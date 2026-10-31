//! Token sampling, matching `sample_from_logits` and `top_k_top_p_filtering` in `model/kronos.py`.

use rand::Rng;

/// Samples one token id from `logits`.
///
/// Like the Python code: logits are divided by `temperature`, then `top_k > 0`
/// keeps the `top_k` highest logits (and `top_p` is ignored); otherwise
/// `top_p < 1` keeps the smallest set of tokens whose probability reaches `top_p`.
/// `top_k = 1` always picks the most likely token, which makes predictions deterministic.
pub fn sample<R: Rng>(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut R) -> u32 {
    let mut logits: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

    if top_k > 0 {
        let k = top_k.min(logits.len());
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[k - 1];
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    } else if top_p < 1.0 {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let sorted: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);
        let mut cumulative = 0.0;
        let mut remove = vec![false; sorted.len()];
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            // Shifted by one so the token that crosses the threshold is kept
            if i + 1 < remove.len() {
                remove[i + 1] = cumulative > top_p;
            }
        }
        for (rank, &idx) in order.iter().enumerate() {
            if remove[rank] {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    let probs = softmax(&logits);
    let mut r: f32 = rng.random::<f32>();
    for (i, p) in probs.iter().enumerate() {
        if r < *p {
            return i as u32;
        }
        r -= p;
    }
    // Rounding left a little probability mass: return the last allowed token
    probs.iter().rposition(|p| *p > 0.0).unwrap_or(0) as u32
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn top_k_one_is_argmax() {
        let mut rng = StdRng::seed_from_u64(0);
        let logits = [0.1, 2.0, -1.0, 1.9];
        for _ in 0..100 {
            assert_eq!(sample(&logits, 1.0, 1, 1.0, &mut rng), 1);
        }
    }

    #[test]
    fn top_p_keeps_the_crossing_token() {
        let mut rng = StdRng::seed_from_u64(0);
        // probabilities roughly 0.64, 0.24, 0.09, 0.03: top_p 0.7 keeps the first two
        let logits = [3.0, 2.0, 1.0, 0.0];
        for _ in 0..1000 {
            assert!(sample(&logits, 1.0, 0, 0.7, &mut rng) < 2);
        }
    }
}
