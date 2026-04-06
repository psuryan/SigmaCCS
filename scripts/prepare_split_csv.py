"""
Utility: given data.csv and a split.json, write out train.csv, val.csv, test.csv
with columns SMILES, Adduct, True CCS.

Usage (standalone):
    python prepare_split_csv.py \
        --data ../GraphCCS/data/data.csv \
        --split ../GraphCCS/data/splits/random/split.json \
        --out /tmp/csvs

Usage (imported):
    from scripts.prepare_split_csv import prepare_split_csvs
    train_csv, val_csv, test_csv = prepare_split_csvs(data_path, split_path, out_dir)
"""

import argparse
import json
import os
import pandas as pd


def prepare_split_csvs(data_path: str, split_path: str, out_dir: str):
    """
    Read data.csv + split.json, write train/val/test CSVs to out_dir.

    Returns (train_csv, val_csv, test_csv) absolute paths.
    """
    df = pd.read_csv(data_path)
    # Normalise column names — data.csv uses: index, smiles, adducts, label
    df = df.rename(columns={
        "smiles": "SMILES",
        "adducts": "Adduct",
        "label": "True CCS",
    })
    needed = ["SMILES", "Adduct", "True CCS"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {data_path}. "
                             f"Available: {list(df.columns)}")

    with open(split_path) as f:
        split = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for subset in ("train", "val", "test"):
        indices = split[subset]
        subset_df = df.iloc[indices][needed].reset_index(drop=True)
        out_path = os.path.join(out_dir, f"{subset}.csv")
        subset_df.to_csv(out_path, index=False)
        paths[subset] = out_path
        print(f"  {subset}: {len(subset_df)} rows -> {out_path}")

    return paths["train"], paths["val"], paths["test"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val/test CSVs from a split.json")
    parser.add_argument("--data", required=True, help="Path to data.csv")
    parser.add_argument("--split", required=True, help="Path to split.json")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    prepare_split_csvs(args.data, args.split, args.out)
