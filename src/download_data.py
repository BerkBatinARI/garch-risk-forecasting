# src/download_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")

# Stooq daily data endpoints (free, no API key)
# Examples:
# https://stooq.com/q/d/l/?s=spy.us&i=d
# https://stooq.com/q/d/l/?s=tlt.us&i=d
TICKERS = {
    "SPY": "spy.us",  # US equities proxy
    "TLT": "tlt.us",  # long duration bonds proxy
    "GLD": "gld.us",  # gold proxy
}


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    stooq_symbol: str


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    # Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # Expect: date, open, high, low, close, volume
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Keep only what we need for risk work
    keep = ["date", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' from Stooq data for {symbol}. Got: {df.columns.tolist()}")

    df = df[keep].rename(columns={"close": "Close", "volume": "Volume", "date": "Date"})
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for name, stooq_symbol in TICKERS.items():
        df = fetch_stooq_daily(stooq_symbol)
        out_path = DATA_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)

        metadata_rows.append(
            {
                "ticker": name,
                "source": "stooq",
                "symbol": stooq_symbol,
                "rows": len(df),
                "start": df["Date"].iloc[0].date().isoformat(),
                "end": df["Date"].iloc[-1].date().isoformat(),
            }
        )
        print(f"Saved {out_path} ({len(df)} rows)")

    meta = pd.DataFrame(metadata_rows)
    meta_path = DATA_DIR / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()