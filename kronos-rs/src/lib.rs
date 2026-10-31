//! Offline Rust inference for Kronos, a foundation model for financial candlesticks.
//!
//! This crate never touches the network: model weights are read from local
//! `safetensors` files, which can only contain numbers, not code.

#![forbid(unsafe_code)]

pub mod data;
pub mod model;
pub mod predictor;
pub mod sampling;
pub mod weights;

pub use data::{Candle, Series, TimeFeatures};
pub use model::{Kronos, KronosConfig, KronosTokenizer, TokenizerConfig};
pub use predictor::{predict, PredictParams};
