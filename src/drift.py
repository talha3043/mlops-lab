"""Data drift check: compare every feature of a new batch with the training data."""
import argparse
import sys

import pandas as pd
from scipy.stats import ks_2samp

from src import ml


def main():
    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--reference", default=str(ml.DATA_PATH))
    parser.add_argument("--current", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="largest allowed share of drifted features")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference).drop(columns=[ml.TARGET], errors="ignore")
    current = pd.read_csv(args.current).drop(columns=[ml.TARGET], errors="ignore")

    print(f"{'feature':<24}{'KS statistic':>14}{'p-value':>12}  result")
    drifted = 0
    for column in reference.columns:
        result = ks_2samp(reference[column], current[column])
        is_drift = result.pvalue < args.alpha
        drifted += int(is_drift)
        print(f"{column:<24}{result.statistic:>14.3f}{result.pvalue:>12.4f}  "
              f"{'DRIFT' if is_drift else 'ok'}")

    share = drifted / len(reference.columns)
    print(f"\nDrifted features: {drifted} of {len(reference.columns)} ({share:.0%}), "
          f"allowed: {args.threshold:.0%}")
    if share > args.threshold:
        print("DRIFT DETECTED: retrain the model with recent data.")
        sys.exit(1)
    print("NO SIGNIFICANT DRIFT")


if __name__ == "__main__":
    main()
