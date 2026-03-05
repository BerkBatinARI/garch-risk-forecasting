# src/run_all.py
from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    # Run from repo root: python src/run_all.py
    run([sys.executable, "src/download_data.py"])
    run([sys.executable, "src/risk_forecast.py"])
    print("\nDone. Check reports/figures/ and README for embedded plots.")


if __name__ == "__main__":
    main()