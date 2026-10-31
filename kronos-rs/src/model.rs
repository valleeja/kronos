//! Kronos tokenizer and predictor, ported from `model/kronos.py` and `model/module.py`.
//!
//! Only the inference path is implemented. Parameter names match the PyTorch
//! checkpoints published on Hugging Face, so the same `model.safetensors` files load here.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, ops::softmax_last_dim, Embedding, Linear, VarBuilder};
use serde::Deserialize;

/// Hyperparameters of `KronosTokenizer` (its `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub d_in: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    pub n_enc_layers: usize,
    pub n_dec_layers: usize,
    pub s1_bits: usize,
    pub s2_bits: usize,
}

/// Hyperparameters of `Kronos` (its `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct KronosConfig {
    pub s1_bits: usize,
    pub s2_bits: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    #[serde(default)]
    pub learn_te: bool,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(mean_sq + self.eps)?.sqrt()?)?;
        x.broadcast_mul(&self.weight)
    }
}

struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(d_model: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(d_model, ff_dim, vb.pp("w1"))?,
            w2: linear_no_bias(ff_dim, d_model, vb.pp("w2"))?,
            w3: linear_no_bias(d_model, ff_dim, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        self.w2.forward(&(gate * self.w3.forward(x)?)?)
    }
}

/// Rotary position tables for `seq_len` positions, shape `[seq_len, head_dim]`.
fn rope_tables(seq_len: usize, head_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / head_dim as f32))
        .collect();
    let half = inv_freq.len();
    let mut emb = Vec::with_capacity(seq_len * head_dim);
    for t in 0..seq_len {
        let row: Vec<f32> = inv_freq.iter().map(|f| t as f32 * f).collect();
        emb.extend_from_slice(&row);
        emb.extend_from_slice(&row);
    }
    let emb = Tensor::from_vec(emb, (seq_len, 2 * half), device)?;
    Ok((emb.cos()?, emb.sin()?))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}

/// `[seq_len, seq_len]` additive mask: 0 on and below the diagonal, -inf above.
fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear(d_model, d_model, vb.pp("q_proj"))?,
            k_proj: linear(d_model, d_model, vb.pp("k_proj"))?,
            v_proj: linear(d_model, d_model, vb.pp("v_proj"))?,
            out_proj: linear(d_model, d_model, vb.pp("out_proj"))?,
            n_heads,
            head_dim: d_model / n_heads,
        })
    }

    fn heads(&self, x: &Tensor, proj: &Linear) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        proj.forward(x)?.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, _, q_len, _) = q.dims4()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let out = softmax_last_dim(&scores)?.matmul(v)?;
        let out = out.transpose(1, 2)?.reshape((b, q_len, self.n_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }

    /// Causal self-attention with rotary positions (`MultiHeadAttentionWithRoPE`).
    fn self_attention(&self, x: &Tensor, rope: &(Tensor, Tensor), mask: &Tensor) -> Result<Tensor> {
        let q = apply_rope(&self.heads(x, &self.q_proj)?, &rope.0, &rope.1)?;
        let k = apply_rope(&self.heads(x, &self.k_proj)?, &rope.0, &rope.1)?;
        let v = self.heads(x, &self.v_proj)?;
        self.attend(&q, &k, &v, Some(mask))
    }

    /// Non-causal cross-attention (`MultiHeadCrossAttentionWithRoPE` in eval mode).
    ///
    /// PyTorch sizes the rotary table by the query length. At inference the
    /// query has length 1, so every position gets the rotation of position 0,
    /// which is the identity: no rotation is applied here.
    fn cross_attention(&self, query: &Tensor, context: &Tensor) -> Result<Tensor> {
        let q = self.heads(query, &self.q_proj)?;
        let k = self.heads(context, &self.k_proj)?;
        let v = self.heads(context, &self.v_proj)?;
        self.attend(&q, &k, &v, None)
    }
}

struct TransformerBlock {
    norm1: RmsNorm,
    self_attn: Attention,
    norm2: RmsNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    fn new(d_model: usize, n_heads: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::new(d_model, vb.pp("norm1"))?,
            self_attn: Attention::new(d_model, n_heads, vb.pp("self_attn"))?,
            norm2: RmsNorm::new(d_model, vb.pp("norm2"))?,
            ffn: FeedForward::new(d_model, ff_dim, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), mask: &Tensor) -> Result<Tensor> {
        let x = (x + self.self_attn.self_attention(&self.norm1.forward(x)?, rope, mask)?)?;
        &x + self.ffn.forward(&self.norm2.forward(&x)?)?
    }
}

fn run_blocks(blocks: &[TransformerBlock], mut x: Tensor, head_dim: usize) -> Result<Tensor> {
    let seq_len = x.dim(1)?;
    let rope = rope_tables(seq_len, head_dim, x.device())?;
    let mask = causal_mask(seq_len, x.device())?;
    for block in blocks {
        x = block.forward(&x, &rope, &mask)?;
    }
    Ok(x)
}

/// Turns candles into hierarchical tokens (s1, s2) and back (`KronosTokenizer`).
pub struct KronosTokenizer {
    pub config: TokenizerConfig,
    embed: Linear,
    head: Linear,
    encoder: Vec<TransformerBlock>,
    decoder: Vec<TransformerBlock>,
    quant_embed: Linear,
    post_quant_embed: Linear,
}

impl KronosTokenizer {
    pub fn new(config: TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let c = &config;
        if c.d_model % c.n_heads != 0 || c.n_enc_layers == 0 || c.n_dec_layers == 0 {
            candle_core::bail!("invalid tokenizer config: {c:?}");
        }
        let codebook_dim = c.s1_bits + c.s2_bits;
        let blocks = |n: usize, name: &str| -> Result<Vec<TransformerBlock>> {
            (0..n - 1)
                .map(|i| TransformerBlock::new(c.d_model, c.n_heads, c.ff_dim, vb.pp(format!("{name}.{i}"))))
                .collect()
        };
        Ok(Self {
            embed: linear(c.d_in, c.d_model, vb.pp("embed"))?,
            head: linear(c.d_model, c.d_in, vb.pp("head"))?,
            encoder: blocks(c.n_enc_layers, "encoder")?,
            decoder: blocks(c.n_dec_layers, "decoder")?,
            quant_embed: linear(c.d_model, codebook_dim, vb.pp("quant_embed"))?,
            post_quant_embed: linear(codebook_dim, c.d_model, vb.pp("post_quant_embed"))?,
            config,
        })
    }

    fn head_dim(&self) -> usize {
        self.config.d_model / self.config.n_heads
    }

    /// Encodes normalized candles `[batch, seq_len, d_in]` into (s1, s2) token ids per batch row.
    ///
    /// Binary spherical quantization keeps only the sign of each code
    /// dimension, so the L2 normalization done in PyTorch does not change the result.
    pub fn encode(&self, x: &Tensor) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let z = run_blocks(&self.encoder, self.embed.forward(x)?, self.head_dim())?;
        let z = self.quant_embed.forward(&z)?.to_vec3::<f32>()?;
        let (s1_bits, s2_bits) = (self.config.s1_bits, self.config.s2_bits);
        let to_index = |code: &[f32]| -> u32 {
            code.iter().enumerate().map(|(i, v)| if *v > 0.0 { 1u32 << i } else { 0 }).sum()
        };
        let mut s1 = Vec::with_capacity(z.len());
        let mut s2 = Vec::with_capacity(z.len());
        for row in &z {
            s1.push(row.iter().map(|code| to_index(&code[..s1_bits])).collect());
            s2.push(row.iter().map(|code| to_index(&code[s1_bits..s1_bits + s2_bits])).collect());
        }
        Ok((s1, s2))
    }

    /// Decodes (s1, s2) token ids, one sequence per batch row, into normalized candles `[batch, seq_len, d_in]`.
    pub fn decode(&self, s1: &[Vec<u32>], s2: &[Vec<u32>], device: &Device) -> Result<Tensor> {
        let (s1_bits, s2_bits) = (self.config.s1_bits, self.config.s2_bits);
        let codebook_dim = s1_bits + s2_bits;
        let scale = 1.0 / (codebook_dim as f32).sqrt();
        let batch = s1.len();
        let seq_len = s1.first().map_or(0, |r| r.len());
        let bit = |id: u32, i: usize| if (id >> i) & 1 == 1 { scale } else { -scale };
        let mut bits = Vec::with_capacity(batch * seq_len * codebook_dim);
        for (row1, row2) in s1.iter().zip(s2) {
            for (&a, &b) in row1.iter().zip(row2) {
                bits.extend((0..s1_bits).map(|i| bit(a, i)));
                bits.extend((0..s2_bits).map(|i| bit(b, i)));
            }
        }
        let quantized = Tensor::from_vec(bits, (batch, seq_len, codebook_dim), device)?;
        let z = run_blocks(&self.decoder, self.post_quant_embed.forward(&quantized)?, self.head_dim())?;
        self.head.forward(&z)
    }
}

/// Sinusoidal table used by `FixedEmbedding` when temporal embeddings are not learned.
fn fixed_embedding(size: usize, d_model: usize, device: &Device) -> Result<Embedding> {
    let mut w = vec![0f32; size * d_model];
    for pos in 0..size {
        for i in (0..d_model).step_by(2) {
            let div = (i as f32 * -(10000f32.ln() / d_model as f32)).exp();
            w[pos * d_model + i] = (pos as f32 * div).sin();
            if i + 1 < d_model {
                w[pos * d_model + i + 1] = (pos as f32 * div).cos();
            }
        }
    }
    Ok(Embedding::new(Tensor::from_vec(w, (size, d_model), device)?, d_model))
}

/// The autoregressive predictor (`Kronos`).
pub struct Kronos {
    pub config: KronosConfig,
    emb_s1: Embedding,
    emb_s2: Embedding,
    fusion_proj: Linear,
    time_embeds: Vec<Embedding>,
    blocks: Vec<TransformerBlock>,
    norm: RmsNorm,
    dep_attn: Attention,
    dep_norm: RmsNorm,
    proj_s1: Linear,
    proj_s2: Linear,
}

/// Number of heads of `DependencyAwareLayer`, fixed in the PyTorch code.
const DEP_LAYER_HEADS: usize = 4;

impl Kronos {
    pub fn new(config: KronosConfig, vb: VarBuilder) -> Result<Self> {
        let c = &config;
        let d = c.d_model;
        if d % c.n_heads != 0 || d % DEP_LAYER_HEADS != 0 {
            candle_core::bail!("invalid model config: {c:?}");
        }
        let te = vb.pp("time_emb");
        let mut time_embeds = Vec::new();
        // Order matches the time features: minute, hour, weekday, day, month
        for (name, size) in [("minute_embed", 60), ("hour_embed", 24), ("weekday_embed", 7), ("day_embed", 32), ("month_embed", 13)] {
            let emb = if c.learn_te {
                embedding(size, d, te.pp(name))?
            } else if te.contains_tensor(&format!("{name}.emb.weight")) {
                embedding(size, d, te.pp(name).pp("emb"))?
            } else {
                fixed_embedding(size, d, vb.device())?
            };
            time_embeds.push(emb);
        }
        let emb = vb.pp("embedding");
        Ok(Self {
            emb_s1: embedding(1 << c.s1_bits, d, emb.pp("emb_s1"))?,
            emb_s2: embedding(1 << c.s2_bits, d, emb.pp("emb_s2"))?,
            fusion_proj: linear(2 * d, d, emb.pp("fusion_proj"))?,
            time_embeds,
            blocks: (0..c.n_layers)
                .map(|i| TransformerBlock::new(d, c.n_heads, c.ff_dim, vb.pp(format!("transformer.{i}"))))
                .collect::<Result<_>>()?,
            norm: RmsNorm::new(d, vb.pp("norm"))?,
            dep_attn: Attention::new(d, DEP_LAYER_HEADS, vb.pp("dep_layer.cross_attn"))?,
            dep_norm: RmsNorm::new(d, vb.pp("dep_layer.norm"))?,
            proj_s1: linear(d, 1 << c.s1_bits, vb.pp("head.proj_s1"))?,
            proj_s2: linear(d, 1 << c.s2_bits, vb.pp("head.proj_s2"))?,
            config,
        })
    }

    /// Predicts the next s1 token.
    ///
    /// `s1`, `s2`: `[batch, seq_len]` u32 ids. `stamp`: `[batch, seq_len, 5]` u32 time features.
    /// Returns the s1 logits of the last position `[batch, vocab_s1]` and the context `[batch, seq_len, d_model]`.
    pub fn decode_s1(&self, s1: &Tensor, s2: &Tensor, stamp: &Tensor) -> Result<(Tensor, Tensor)> {
        let scale = (self.config.d_model as f64).sqrt();
        let e1 = (self.emb_s1.forward(s1)? * scale)?;
        let e2 = (self.emb_s2.forward(s2)? * scale)?;
        let mut x = self.fusion_proj.forward(&Tensor::cat(&[&e1, &e2], D::Minus1)?)?;
        for (i, emb) in self.time_embeds.iter().enumerate() {
            let ids = stamp.narrow(D::Minus1, i, 1)?.squeeze(D::Minus1)?.contiguous()?;
            x = (x + emb.forward(&ids)?)?;
        }
        let x = run_blocks(&self.blocks, x, self.config.d_model / self.config.n_heads)?;
        let context = self.norm.forward(&x)?;
        let last = context.narrow(1, context.dim(1)? - 1, 1)?.squeeze(1)?;
        Ok((self.proj_s1.forward(&last)?, context))
    }

    /// Predicts the s2 token of the last position, given its s1 token `[batch]`.
    /// Returns `[batch, vocab_s2]` logits.
    pub fn decode_s2(&self, context: &Tensor, s1_last: &Tensor) -> Result<Tensor> {
        let sibling = self.emb_s1.forward(s1_last)?.unsqueeze(1)?;
        let attn = self.dep_attn.cross_attention(&sibling, context)?.squeeze(1)?;
        let last = context.narrow(1, context.dim(1)? - 1, 1)?.squeeze(1)?;
        self.proj_s2.forward(&self.dep_norm.forward(&(last + attn)?)?)
    }
}

/// Weights are loaded and computed in 32-bit floats, like the PyTorch reference.
pub const DTYPE: DType = DType::F32;
