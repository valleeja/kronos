//! Candle data: a small, strict CSV reader and the calendar features Kronos uses.

use std::fmt;

/// One candle: open, high, low, close, volume, amount.
pub type Candle = [f32; 6];

/// Calendar features fed to the model: minute, hour, weekday (Monday = 0), day, month.
pub type TimeFeatures = [u32; 5];

/// A date and time, without time zone, to the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DateTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 }, m, d)
}

impl DateTime {
    /// Parses `YYYY-MM-DD`, `YYYY/MM/DD`, optionally followed by ` ` or `T` and `HH:MM[:SS[.fff]]`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (date, time) = match s.find([' ', 'T']) {
            Some(i) => (&s[..i], s[i + 1..].trim()),
            None => (s, ""),
        };
        let mut d = date.split(['-', '/']);
        let year: i64 = d.next()?.parse().ok()?;
        let month: u32 = d.next()?.parse().ok()?;
        let day: u32 = d.next()?.parse().ok()?;
        if d.next().is_some() {
            return None;
        }
        let (mut hour, mut minute, mut second) = (0, 0, 0);
        if !time.is_empty() {
            let mut t = time.split(':');
            hour = t.next()?.parse().ok()?;
            minute = t.next()?.parse().ok()?;
            if let Some(sec) = t.next() {
                second = sec.split('.').next()?.parse().ok()?;
            }
            if t.next().is_some() {
                return None;
            }
        }
        let dt = Self { year, month, day, hour, minute, second };
        dt.is_valid().then_some(dt)
    }

    fn is_valid(&self) -> bool {
        (1..=12).contains(&self.month)
            && self.day >= 1
            && civil_from_days(days_from_civil(self.year, self.month, self.day)) == (self.year, self.month, self.day)
            && self.hour < 24
            && self.minute < 60
            && self.second < 60
    }

    pub fn to_epoch_seconds(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day) * 86_400
            + (self.hour * 3600 + self.minute * 60 + self.second) as i64
    }

    pub fn from_epoch_seconds(secs: i64) -> Self {
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400) as u32;
        let (year, month, day) = civil_from_days(days);
        Self { year, month, day, hour: rem / 3600, minute: rem % 3600 / 60, second: rem % 60 }
    }

    /// Calendar features, computed like `calc_time_stamps` in `model/kronos.py`.
    pub fn features(&self) -> TimeFeatures {
        let weekday = (days_from_civil(self.year, self.month, self.day) + 3).rem_euclid(7) as u32;
        [self.minute, self.hour, weekday, self.day, self.month]
    }
}

impl fmt::Display for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// A candle series read from CSV.
#[derive(Debug, Clone)]
pub struct Series {
    pub timestamps: Vec<DateTime>,
    pub candles: Vec<Candle>,
}

/// Hard limits that protect against oversized or hostile input files.
pub const MAX_CSV_BYTES: usize = 200 * 1024 * 1024;
pub const MAX_ROWS: usize = 5_000_000;

impl Series {
    /// Parses CSV text with a header row.
    ///
    /// Required columns: a timestamp (`timestamps`, `timestamp`, `date`, `datetime` or `time`),
    /// `open`, `high`, `low`, `close`. Optional: `volume` (`vol`) and `amount` (`amt`).
    /// Like the Python predictor, a missing volume sets volume and amount to 0, and a
    /// missing amount is estimated as volume times the mean of open, high, low and close.
    pub fn from_csv(text: &str) -> Result<Self, String> {
        if text.len() > MAX_CSV_BYTES {
            return Err(format!("CSV file is larger than {} MB", MAX_CSV_BYTES / 1024 / 1024));
        }
        if text.contains('"') {
            return Err("quoted CSV fields are not supported".into());
        }
        let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
        let (_, header) = lines.next().ok_or("CSV file is empty")?;
        let names: Vec<String> =
            header.trim_start_matches('\u{feff}').split(',').map(|h| h.trim().to_lowercase()).collect();
        let find = |aliases: &[&str]| names.iter().position(|n| aliases.contains(&n.as_str()));
        let ts_col = find(&["timestamps", "timestamp", "date", "datetime", "time"]).ok_or("missing timestamp column")?;
        let price_cols = ["open", "high", "low", "close"]
            .iter()
            .map(|c| find(&[c]).ok_or(format!("missing column '{c}'")))
            .collect::<Result<Vec<_>, _>>()?;
        let vol_col = find(&["volume", "vol"]);
        let amt_col = find(&["amount", "amt"]);

        let mut series = Series { timestamps: Vec::new(), candles: Vec::new() };
        for (line_no, line) in lines {
            if series.candles.len() >= MAX_ROWS {
                return Err(format!("CSV file has more than {MAX_ROWS} rows"));
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let row = line_no + 1;
            let get = |col: usize| -> Result<f32, String> {
                let raw = fields.get(col).ok_or(format!("line {row}: missing field"))?;
                let v: f32 = raw.parse().map_err(|_| format!("line {row}: '{raw}' is not a number"))?;
                if v.is_finite() { Ok(v) } else { Err(format!("line {row}: value is not finite")) }
            };
            let raw_ts = fields.get(ts_col).ok_or(format!("line {row}: missing timestamp"))?;
            let ts = DateTime::parse(raw_ts).ok_or(format!("line {row}: cannot read timestamp '{raw_ts}'"))?;
            let p: Vec<f32> = price_cols.iter().map(|&c| get(c)).collect::<Result<_, _>>()?;
            let (volume, amount) = match (vol_col, amt_col) {
                (None, _) => (0.0, 0.0),
                (Some(v), Some(a)) => (get(v)?, get(a)?),
                (Some(v), None) => {
                    let volume = get(v)?;
                    (volume, volume * (p[0] + p[1] + p[2] + p[3]) / 4.0)
                }
            };
            series.timestamps.push(ts);
            series.candles.push([p[0], p[1], p[2], p[3], volume, amount]);
        }
        if series.candles.is_empty() {
            return Err("CSV file has no data rows".into());
        }
        Ok(series)
    }

    /// Timestamps that continue the series at its usual spacing (the most common
    /// gap among the last 100 rows). Market closures are not skipped.
    pub fn extrapolate_timestamps(&self, end: usize, count: usize) -> Result<Vec<DateTime>, String> {
        let ts = &self.timestamps[..end];
        if ts.len() < 2 {
            return Err("need at least 2 rows to infer the time step".into());
        }
        let start = ts.len().saturating_sub(101);
        let mut gaps: Vec<i64> =
            ts[start..].windows(2).map(|w| w[1].to_epoch_seconds() - w[0].to_epoch_seconds()).collect();
        gaps.sort_unstable();
        let mut step = gaps[0];
        let mut best = 0;
        for chunk in gaps.chunk_by(|a, b| a == b) {
            if chunk.len() > best {
                best = chunk.len();
                step = chunk[0];
            }
        }
        if step <= 0 {
            return Err("timestamps must be increasing".into());
        }
        let last = ts[ts.len() - 1].to_epoch_seconds();
        Ok((1..=count as i64).map(|i| DateTime::from_epoch_seconds(last + i * step)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_formats() {
        let a = DateTime::parse("2019/11/26 9:35").unwrap();
        assert_eq!((a.year, a.month, a.day, a.hour, a.minute), (2019, 11, 26, 9, 35));
        let b = DateTime::parse("2024-02-29T23:59:59.123").unwrap();
        assert_eq!((b.day, b.second), (29, 59));
        assert!(DateTime::parse("2023-02-29").is_none());
        assert!(DateTime::parse("not a date").is_none());
    }

    #[test]
    fn weekday_matches_pandas() {
        // 2025-09-19 is a Friday: pandas weekday 4
        assert_eq!(DateTime::parse("2025-09-19 10:05").unwrap().features(), [5, 10, 4, 19, 9]);
        assert_eq!(DateTime::parse("1970-01-05").unwrap().features()[2], 0);
    }

    #[test]
    fn epoch_round_trip() {
        let dt = DateTime::parse("2001-03-04 05:06:07").unwrap();
        assert_eq!(DateTime::from_epoch_seconds(dt.to_epoch_seconds()), dt);
    }

    #[test]
    fn reads_csv_and_fills_amount() {
        let s = Series::from_csv("\u{feff}timestamps,open,close,high,low,volume\n2024-01-01 09:30,1,2,3,0,10\n").unwrap();
        assert_eq!(s.candles[0], [1.0, 3.0, 0.0, 2.0, 10.0, 15.0]);
        assert!(Series::from_csv("timestamps,open,high,low,close\n2024-01-01,1,2,NaN,1\n").is_err());
    }
}
