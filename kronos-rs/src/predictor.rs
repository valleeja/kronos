//! End-to-end forecasting, ported from `KronosPredictor.predict` and
//! `auto_regressive_inference` in `model/kronos.py`.

use candle_core::{Device, Result, Tensor};
use rand::{rngs::StdRng, SeedableRng};

use crate::data::{Candle, TimeFeatures};
use crate::model::{Kronos, KronosTokenizer};
use crate::sampling::sample;

#[derive(Debug, Clone)]
pub struct PredictParams {
    /// Number of future candles to generate.
    pub pred_len: usize,
    /// Longest token window the model sees (512 for Kronos-small and base, 2048 for mini).
    pub max_context: usize,
    /// Normalized values are clipped to `[-clip, clip]` standard deviations.
    pub clip: f32,
    pub temperature: f32,
    /// `1` makes predictions deterministic; `0` disables top-k.
    pub top_k: usize,
    pub top_p: f32,
    /// Number of sampled paths, averaged into the final forecast.
    pub sample_count: usize,
    pub seed: u64,
}

impl Default for PredictParams {
    fn default() -> Self {
        Self { pred_len: 120, max_context: 512, clip: 5.0, temperature: 1.0, top_k: 0, top_p: 0.9, sample_count: 1, seed: 42 }
    }
}

/// Forecasts `params.pred_len` candles after `history`.
///
/// `history_time` and `future_time` hold the calendar features of the history
/// and of the candles to predict. Prices are normalized per column on the
/// history window, as in the Python predictor, and the result is returned in
/// the original scale.
pub fn predict(
    tokenizer: &KronosTokenizer,
    model: &Kronos,
    history: &[Candle],
    history_time: &[TimeFeatures],
    future_time: &[TimeFeatures],
    params: &PredictParams,
    device: &Device,
) -> Result<Vec<Candle>> {
    let p = params;
    let n = history.len();
    if n == 0 || history_time.len() != n || future_time.len() != p.pred_len {
        candle_core::bail!("history, history_time and future_time lengths do not match");
    }
    if p.pred_len == 0 || p.pred_len > p.max_context || p.sample_count == 0 {
        candle_core::bail!("pred_len must be in 1..=max_context and sample_count at least 1");
    }

    // Per-column mean and population standard deviation over the history
    let mut mean = [0f64; 6];
    let mut std = [0f64; 6];
    for c in 0..6 {
        mean[c] = history.iter().map(|r| r[c] as f64).sum::<f64>() / n as f64;
        std[c] = (history.iter().map(|r| (r[c] as f64 - mean[c]).powi(2)).sum::<f64>() / n as f64).sqrt();
    }
    let normalized: Vec<f32> = history
        .iter()
        .flat_map(|r| (0..6).map(move |c| r[c] as f64))
        .enumerate()
        .map(|(i, v)| (((v - mean[i % 6]) / (std[i % 6] + 1e-5)) as f32).clamp(-p.clip, p.clip))
        .collect();

    // Every sampled path starts from the same tokens, so encode once
    let x = Tensor::from_vec(normalized, (1, n, 6), device)?;
    let (s1, s2) = tokenizer.encode(&x)?;
    let mut paths_s1 = vec![s1[0].clone(); p.sample_count];
    let mut paths_s2 = vec![s2[0].clone(); p.sample_count];
    let stamps: Vec<TimeFeatures> = history_time.iter().chain(future_time).copied().collect();

    let mut rng = StdRng::seed_from_u64(p.seed);
    let batch = p.sample_count;
    for step in 0..p.pred_len {
        let len = n + step;
        let window = len.min(p.max_context);
        let start = len - window;
        let ids = |paths: &[Vec<u32>]| -> Result<Tensor> {
            let flat: Vec<u32> = paths.iter().flat_map(|path| path[start..len].iter().copied()).collect();
            Tensor::from_vec(flat, (batch, window), device)
        };
        let stamp: Vec<u32> = (0..batch).flat_map(|_| stamps[start..len].iter().flatten().copied()).collect();
        let stamp = Tensor::from_vec(stamp, (batch, window, 5), device)?;

        let (s1_logits, context) = model.decode_s1(&ids(&paths_s1)?, &ids(&paths_s2)?, &stamp)?;
        let s1_next: Vec<u32> = s1_logits
            .to_vec2::<f32>()?
            .iter()
            .map(|l| sample(l, p.temperature, p.top_k, p.top_p, &mut rng))
            .collect();
        let s2_logits = model.decode_s2(&context, &Tensor::new(s1_next.as_slice(), device)?)?;
        let s2_next: Vec<u32> = s2_logits
            .to_vec2::<f32>()?
            .iter()
            .map(|l| sample(l, p.temperature, p.top_k, p.top_p, &mut rng))
            .collect();
        for b in 0..batch {
            paths_s1[b].push(s1_next[b]);
            paths_s2[b].push(s2_next[b]);
        }
    }

    // Decode the last window of each path back to candles and average the paths
    let total = n + p.pred_len;
    let start = total - total.min(p.max_context);
    let tail = |paths: &[Vec<u32>]| -> Vec<Vec<u32>> { paths.iter().map(|path| path[start..].to_vec()).collect() };
    let decoded = tokenizer.decode(&tail(&paths_s1), &tail(&paths_s2), device)?.to_vec3::<f32>()?;
    let window = total - start;
    let mut forecast = Vec::with_capacity(p.pred_len);
    for t in window - p.pred_len..window {
        let mut candle = [0f32; 6];
        for (c, value) in candle.iter_mut().enumerate() {
            let avg = decoded.iter().map(|path| path[t][c] as f64).sum::<f64>() / batch as f64;
            *value = (avg * (std[c] + 1e-5) + mean[c]) as f32;
        }
        forecast.push(candle);
    }
    Ok(forecast)
}
