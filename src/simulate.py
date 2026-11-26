"""Create a batch of 'new' data, with or without a shift, to test drift detection."""
import argparse

import numpy as np

from src import ml

SHIFTED = ["alcohol", "flavanoids", "color_intensity", "proline", "magnesium", "hue"]


def main():
    parser = argparse.ArgumentParser(description="Simulate a new batch of data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shift", type=float, default=0.0,
                        help="how many standard deviations to move 6 features")
    parser.add_argument("--rows", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    X, y = ml.load_data()
    batch = X.sample(n=args.rows, random_state=7).reset_index(drop=True)
    for column in batch.columns:
        std = X[column].std()
        batch[column] = batch[column] + rng.normal(0, 0.01 * std, size=len(batch))
        if column in SHIFTED:
            batch[column] = batch[column] + args.shift * std
    batch.to_csv(args.out, index=False)
    print(f"Wrote {len(batch)} rows to {args.out} (shift = {args.shift})")


if __name__ == "__main__":
    main()
