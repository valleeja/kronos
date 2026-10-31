//! `kronos` command line: forecast candles from a CSV file, fully offline.

#![forbid(unsafe_code)]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::Device;
use kronos::data::{DateTime, Series};
use kronos::weights::{load_model, load_tokenizer};
use kronos::{predict, PredictParams};

const USAGE: &str = "\
kronos: forecast financial candles with Kronos, offline

USAGE:
    kronos --model DIR --tokenizer DIR --input FILE.csv [OPTIONS]

REQUIRED:
    --model DIR            Folder with config.json and model.safetensors (e.g. Kronos-small)
    --tokenizer DIR        Folder with config.json and model.safetensors (e.g. Kronos-Tokenizer-base)
    --input FILE           CSV with timestamps,open,high,low,close[,volume[,amount]]

OPTIONS:
    --output FILE          Write the forecast CSV here instead of the screen
    --lookback N           History candles fed to the model [default: 400]
    --pred-len N           Candles to forecast [default: 120]
    --context-end N        Backtest: use the N rows before row N as history and the
                           next rows of the file as the future, printing the real close
                           next to the forecast. Default: forecast after the last row.
    --max-context N        Token window of the model: 512 for small/base, 2048 for mini [default: 512]
    --temperature T        Sampling temperature [default: 1.0]
    --top-k K              Keep the K most likely tokens; 1 = deterministic [default: 0]
    --top-p P              Nucleus sampling threshold [default: 0.9]
    --samples N            Paths to sample and average [default: 1]
    --seed N               Random seed [default: 42]
    --model-sha256 HEX     Refuse to run unless the model weights have this SHA-256
    --tokenizer-sha256 HEX Refuse to run unless the tokenizer weights have this SHA-256
    -h, --help             Show this help
";

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    lookback: usize,
    context_end: Option<usize>,
    model_sha256: Option<String>,
    tokenizer_sha256: Option<String>,
    params: PredictParams,
}

// Upper bounds on user-supplied sizes, to keep runtime and memory predictable
const MAX_LOOKBACK: usize = 8192;
const MAX_PRED_LEN: usize = 2048;
const MAX_SAMPLES: usize = 64;

fn parse_args() -> Result<Option<Args>, String> {
    let mut raw = std::env::args().skip(1);
    let (mut model, mut tokenizer, mut input, mut output) = (None, None, None, None);
    let (mut lookback, mut context_end) = (400usize, None);
    let (mut model_sha256, mut tokenizer_sha256) = (None, None);
    let mut params = PredictParams::default();
    while let Some(flag) = raw.next() {
        if flag == "-h" || flag == "--help" {
            return Ok(None);
        }
        let value = raw.next().ok_or(format!("{flag} needs a value"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("{flag}: '{v}' is not a whole number"));
        let float = |v: &str| v.parse::<f32>().map_err(|_| format!("{flag}: '{v}' is not a number"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--lookback" => lookback = num(&value)?,
            "--pred-len" => params.pred_len = num(&value)?,
            "--context-end" => context_end = Some(num(&value)?),
            "--max-context" => params.max_context = num(&value)?,
            "--temperature" => params.temperature = float(&value)?,
            "--top-k" => params.top_k = num(&value)?,
            "--top-p" => params.top_p = float(&value)?,
            "--samples" => params.sample_count = num(&value)?,
            "--seed" => params.seed = value.parse().map_err(|_| format!("--seed: '{value}' is not a whole number"))?,
            "--model-sha256" => model_sha256 = Some(value),
            "--tokenizer-sha256" => tokenizer_sha256 = Some(value),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let p = &params;
    if !(1..=MAX_LOOKBACK).contains(&lookback) {
        return Err(format!("--lookback must be between 1 and {MAX_LOOKBACK}"));
    }
    if !(1..=MAX_PRED_LEN).contains(&p.pred_len) || p.pred_len > p.max_context {
        return Err(format!("--pred-len must be between 1 and {MAX_PRED_LEN}, and at most --max-context"));
    }
    if !(1..=MAX_SAMPLES).contains(&p.sample_count) {
        return Err(format!("--samples must be between 1 and {MAX_SAMPLES}"));
    }
    if !(p.temperature > 0.0 && p.temperature <= 10.0) || !(p.top_p > 0.0 && p.top_p <= 1.0) {
        return Err("--temperature must be in (0, 10] and --top-p in (0, 1]".into());
    }
    if !(1..=8192).contains(&p.max_context) {
        return Err("--max-context must be between 1 and 8192".into());
    }
    Ok(Some(Args {
        model: model.ok_or("--model is required")?,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        input: input.ok_or("--input is required")?,
        output,
        lookback,
        context_end,
        model_sha256,
        tokenizer_sha256,
        params,
    }))
}

fn run(args: Args) -> Result<(), String> {
    let device = Device::Cpu;
    let (tokenizer, tok_digest) = load_tokenizer(&args.tokenizer, args.tokenizer_sha256.as_deref(), &device)?;
    let (model, model_digest) = load_model(&args.model, args.model_sha256.as_deref(), &device)?;
    eprintln!("tokenizer sha256: {tok_digest}");
    eprintln!("model     sha256: {model_digest}");

    let meta = fs::metadata(&args.input).map_err(|e| format!("{}: {e}", args.input.display()))?;
    if meta.len() as usize > kronos::data::MAX_CSV_BYTES {
        return Err(format!("{}: file is too large", args.input.display()));
    }
    let text = fs::read_to_string(&args.input).map_err(|e| format!("{}: {e}", args.input.display()))?;
    let series = Series::from_csv(&text).map_err(|e| format!("{}: {e}", args.input.display()))?;
    let rows = series.candles.len();
    let pred_len = args.params.pred_len;

    let end = args.context_end.unwrap_or(rows);
    if end > rows || end < args.lookback {
        return Err(format!("not enough rows: need {} rows of history before row {end}, file has {rows}", args.lookback));
    }
    let start = end - args.lookback;
    let (future, actual): (Vec<DateTime>, Option<&[kronos::Candle]>) = match args.context_end {
        Some(_) if end + pred_len <= rows => (series.timestamps[end..end + pred_len].to_vec(), Some(&series.candles[end..end + pred_len])),
        Some(_) => return Err(format!("--context-end {end} + --pred-len {pred_len} goes past the end of the file ({rows} rows)")),
        None => (series.extrapolate_timestamps(end, pred_len)?, None),
    };
    let history_time: Vec<_> = series.timestamps[start..end].iter().map(DateTime::features).collect();
    let future_time: Vec<_> = future.iter().map(DateTime::features).collect();

    eprintln!("forecasting {pred_len} candles from {} history rows ({} to {})", args.lookback, series.timestamps[start], series.timestamps[end - 1]);
    let forecast = predict(&tokenizer, &model, &series.candles[start..end], &history_time, &future_time, &args.params, &device)
        .map_err(|e| format!("prediction failed: {e}"))?;

    let mut out = String::from("timestamp,open,high,low,close,volume,amount");
    if actual.is_some() {
        out.push_str(",actual_close");
    }
    out.push('\n');
    for (i, (ts, c)) in future.iter().zip(&forecast).enumerate() {
        out.push_str(&format!("{ts},{},{},{},{},{},{}", c[0], c[1], c[2], c[3], c[4], c[5]));
        if let Some(actual) = actual {
            out.push_str(&format!(",{}", actual[i][3]));
        }
        out.push('\n');
    }
    match &args.output {
        Some(path) => fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))?,
        None => std::io::stdout().write_all(out.as_bytes()).map_err(|e| e.to_string())?,
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(None) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
