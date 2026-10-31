# kronos-rs

A Rust port of the Kronos inference code (`model/`), built on [candle](https://github.com/huggingface/candle).
It loads the same `model.safetensors` checkpoints as the Python version and forecasts candles from a CSV file.

## Why

- **No Python, pip or PyTorch** at run time: one compiled binary.
- **Never touches the network.** It only reads local model folders. Download them once, then work offline.
- **No `unsafe` code** in this crate (`#![forbid(unsafe_code)]`). Weights are read into memory (not memory-mapped).
- **Weight pinning:** `--model-sha256` and `--tokenizer-sha256` refuse to run if a file was changed. The SHA-256 of the files in use is printed on every run.
- **Bounded inputs:** limits on CSV size (200 MB), rows, weight size (2 GB), lookback, forecast length and samples. Non-finite numbers and malformed timestamps are rejected.
- Usable as a library (`kronos` crate) from another Rust project.

Only inference is ported. Training and fine-tuning stay in Python.

## Build

```bash
cd kronos-rs
cargo build --release --locked
```

The binary is `target/release/kronos`.

## Get the models

Download each folder once from Hugging Face (free, no account needed). Each holds `config.json` and `model.safetensors`:

| Model | Tokenizer | `--max-context` |
|---|---|---|
| [Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini) | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k) | 2048 |
| [Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512 |
| [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512 |

For example, with git:

```bash
git lfs install
git clone https://huggingface.co/NeoQuasar/Kronos-small models/Kronos-small
git clone https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base models/Kronos-Tokenizer-base
```

## Run

Forecast the 120 candles after the end of a file:

```bash
target/release/kronos \
  --model models/Kronos-small --tokenizer models/Kronos-Tokenizer-base \
  --input ../finetune_csv/data/HK_ali_09988_kline_5min_all.csv \
  --lookback 400 --pred-len 120 --output forecast.csv
```

Backtest on history: `--context-end N` uses the 400 rows before row N as history, takes the future timestamps from the file and adds the real close to the output.

Without `--context-end`, future timestamps continue the file at its usual spacing; market closures are not skipped.

`--top-k 1` makes the forecast deterministic. The default (`--top-p 0.9`, one sample) matches the Python defaults; use `--samples N` to average several sampled paths.

Run `kronos --help` for every option.

## Checking the port against Python

`verify/compare.py` runs the Python predictor and this binary in deterministic mode on the same data and compares the forecasts:

```bash
pip install -r requirements.txt
cargo build --release --manifest-path kronos-rs/Cargo.toml
python kronos-rs/verify/compare.py --random                      # small random models, no download
python kronos-rs/verify/compare.py --model models/Kronos-small --tokenizer models/Kronos-Tokenizer-base
```
