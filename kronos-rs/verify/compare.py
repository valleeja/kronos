"""Check that the Rust port gives the same forecast as the Python reference.

Both sides run in deterministic mode (top_k=1, always the most likely token),
so any difference beyond float rounding means the port is wrong.

Usage, from the repository root, with the Python requirements installed:

    # Real models (folders downloaded from Hugging Face)
    python kronos-rs/verify/compare.py --model path/to/Kronos-small --tokenizer path/to/Kronos-Tokenizer-base

    # Small random models, no download needed (checks the code, not the weights)
    python kronos-rs/verify/compare.py --random
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

DEFAULT_CSV = ROOT / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv"
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def make_random_models(out_dir: Path, learn_te: bool, seed: int):
    torch.manual_seed(seed)
    tokenizer = KronosTokenizer(d_in=6, d_model=64, n_heads=4, ff_dim=128, n_enc_layers=3, n_dec_layers=3,
                                ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=6, s2_bits=6,
                                beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
    model = Kronos(s1_bits=6, s2_bits=6, n_layers=2, d_model=64, n_heads=4, ff_dim=128, ffn_dropout_p=0.0,
                   attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=learn_te)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    model.save_pretrained(out_dir / "model")
    return out_dir / "model", out_dir / "tokenizer"


def python_forecast(model_dir, tokenizer_dir, df, lookback, end, pred_len, max_context):
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).eval()
    model = Kronos.from_pretrained(str(model_dir)).eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=max_context)
    x = df.iloc[end - lookback:end]
    y_ts = df["timestamps"].iloc[end:end + pred_len]
    pred = predictor.predict(df=x[COLUMNS].reset_index(drop=True), x_timestamp=x["timestamps"].reset_index(drop=True),
                             y_timestamp=y_ts.reset_index(drop=True), pred_len=pred_len, T=1.0, top_k=1,
                             sample_count=1, verbose=False)
    return pred[COLUMNS].to_numpy()


def rust_forecast(binary, model_dir, tokenizer_dir, csv, lookback, end, pred_len, max_context, out):
    cmd = [str(binary), "--model", str(model_dir), "--tokenizer", str(tokenizer_dir), "--input", str(csv),
           "--lookback", str(lookback), "--context-end", str(end), "--pred-len", str(pred_len),
           "--max-context", str(max_context), "--top-k", "1", "--output", str(out)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(out)[COLUMNS].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--random", action="store_true", help="use small random models instead of real ones")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--binary", default=str(ROOT / "kronos-rs" / "target" / "release" / "kronos"))
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--pred-len", type=int, default=120)
    ap.add_argument("--end", type=int, default=2000, help="row where the forecast starts")
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    work = Path(tempfile.mkdtemp(prefix="kronos-verify-"))

    if args.random:
        cases = [make_random_models(work / f"te{int(te)}", te, seed) for seed, te in [(0, False), (1, True)]]
    elif args.model and args.tokenizer:
        cases = [(Path(args.model), Path(args.tokenizer))]
    else:
        ap.error("give --model and --tokenizer, or --random")

    ok = True
    for model_dir, tokenizer_dir in cases:
        py = python_forecast(model_dir, tokenizer_dir, df, args.lookback, args.end, args.pred_len, args.max_context)
        rs = rust_forecast(args.binary, model_dir, tokenizer_dir, args.csv, args.lookback, args.end, args.pred_len,
                           args.max_context, work / "rust.csv")
        scale = np.abs(py).max(axis=0) + 1e-9
        rel = (np.abs(py - rs) / scale).max()
        print(f"{model_dir}: max difference {rel:.2e} of each column's range")
        ok &= bool(rel <= args.rtol)
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
