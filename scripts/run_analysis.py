"""
Aggregate metrics.json files across seeds and produce:
  1. Summary tables (mean ± std) per split
  2. Learning curve plots for Exp D and Exp E
  3. Per-adduct breakdown tables

Usage:
    python scripts/run_analysis.py --exp-dir experiments [--out-dir results]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available — skipping plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_metrics(exp_dir: str, seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    """Load metrics.json for each seed directory that exists."""
    records = []
    for seed in seeds:
        path = os.path.join(exp_dir, f"seed_{seed}", "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                records.append(json.load(f))
        else:
            print(f"  [missing] {path}")
    return records


def summarise(records: list[dict], subset: str = "test") -> dict:
    """Mean ± std across seeds for one subset (train/val/test)."""
    if not records:
        return {}
    metrics = ("rmse", "mean_pct_diff", "pearson_r", "spearman_r", "kendall_tau")
    result = {}
    for m in metrics:
        vals = [r[subset][m] for r in records if subset in r and m in r[subset]]
        if vals:
            result[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return result


def fmt(summary: dict, metric: str) -> str:
    if metric not in summary:
        return "—"
    m, s = summary[metric]["mean"], summary[metric]["std"]
    if metric in ("pearson_r", "spearman_r", "kendall_tau"):
        return f"{m:.4f} ± {s:.4f}"
    if metric == "mean_pct_diff":
        return f"{m:.4f} ± {s:.4f}%"
    return f"{m:.4f} ± {s:.4f}"


def per_adduct_summary(records: list[dict], subset: str = "test") -> pd.DataFrame:
    """Collect per-adduct RMSE and Mean%Diff across seeds."""
    rows = []
    for r in records:
        if subset not in r:
            continue
        for adduct, vals in r[subset].get("per_adduct", {}).items():
            rows.append({
                "adduct": adduct,
                "rmse": vals.get("rmse"),
                "mean_pct_diff": vals.get("mean_pct_diff"),
                "n": vals.get("n"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("adduct").agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mpd_mean=("mean_pct_diff", "mean"),
        mpd_std=("mean_pct_diff", "std"),
        n=("n", "first"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# Main splits table
# ---------------------------------------------------------------------------

MAIN_SPLITS = {
    "random":             "Exp A — Random",
    "scaffold":           "Exp B — Scaffold",
    "adduct_sensitive":   "Exp C — Adduct-sensitive",
}

METRICS_COLS = [
    ("rmse",          "RMSE"),
    ("mean_pct_diff", "Mean%Diff"),
    ("pearson_r",     "Pearson R"),
    ("spearman_r",    "Spearman R"),
    ("kendall_tau",   "Kendall τ"),
]


def build_main_table(exp_dir: str, seeds) -> pd.DataFrame:
    rows = []
    for split_name, label in MAIN_SPLITS.items():
        d = os.path.join(exp_dir, split_name)
        records = load_seed_metrics(d, seeds)
        if not records:
            print(f"  No results for {split_name}")
            continue
        s = summarise(records, "test")
        row = {"Split": label, "n_seeds": len(records)}
        for key, col in METRICS_COLS:
            row[col] = fmt(s, key)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

FRAC_LABELS = ["frac_0.1", "frac_0.2", "frac_0.4", "frac_0.6", "frac_0.8", "full"]
FRAC_X      = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# GraphCCS reference values for random learning curve
GRAPHCCS_REF = {
    "frac_0.1": {"rmse": 6.815, "mean_pct_diff": 2.606, "pearson_r": 0.9921},
    "frac_0.2": {"rmse": 6.140, "mean_pct_diff": 2.339, "pearson_r": 0.9936},
    "frac_0.4": {"rmse": 5.596, "mean_pct_diff": 2.113, "pearson_r": 0.9948},
    "frac_0.6": {"rmse": 5.315, "mean_pct_diff": 2.008, "pearson_r": 0.9952},
    "frac_0.8": {"rmse": 5.078, "mean_pct_diff": 1.906, "pearson_r": 0.9957},
    "full":     {"rmse": 4.953, "mean_pct_diff": 1.865, "pearson_r": 0.9959},
}


def build_lc_data(exp_dir: str, frac_subdir: str, seeds):
    """Return list of (frac_x, mean, std) for each metric."""
    data = {m: ([], [], []) for m in ("rmse", "mean_pct_diff", "pearson_r")}
    xs = []
    for label, x in zip(FRAC_LABELS, FRAC_X):
        d = os.path.join(exp_dir, frac_subdir, label)
        records = load_seed_metrics(d, seeds)
        if not records:
            continue
        s = summarise(records, "test")
        xs.append(x)
        for m in data:
            if m in s:
                data[m][0].append(x)
                data[m][1].append(s[m]["mean"])
                data[m][2].append(s[m]["std"])
    return data


def build_lc_table(exp_dir: str, frac_subdir: str, seeds) -> pd.DataFrame:
    rows = []
    for label, x in zip(FRAC_LABELS, FRAC_X):
        d = os.path.join(exp_dir, frac_subdir, label)
        records = load_seed_metrics(d, seeds)
        if not records:
            continue
        s = summarise(records, "test")
        row = {"Fraction": label, "x": x, "n_seeds": len(records)}
        for key, col in METRICS_COLS:
            row[col] = fmt(s, key)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_curve(data: dict, ref: dict, out_path: str, title: str):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_info = [
        ("rmse",          "RMSE (Å²)",  ref, "RMSE"),
        ("mean_pct_diff", "Mean%Diff",  ref, "mean_pct_diff"),
        ("pearson_r",     "Pearson R",  ref, "pearson_r"),
    ]
    ref_keys = {"rmse": "rmse", "mean_pct_diff": "mean_pct_diff", "pearson_r": "pearson_r"}

    for ax, (m, ylabel, ref_dict, _) in zip(axes, metric_info):
        xs, means, stds = data[m]
        if not xs:
            continue
        xs_arr = np.array(xs)
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(xs_arr, means_arr, "o-", label="SigmaCCS", color="tab:blue")
        ax.fill_between(xs_arr, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.25, color="tab:blue")

        # GraphCCS reference
        ref_xs, ref_ys = [], []
        for label, x in zip(FRAC_LABELS, FRAC_X):
            if label in ref_dict and ref_keys[m] in ref_dict[label]:
                ref_xs.append(x)
                ref_ys.append(ref_dict[label][ref_keys[m]])
        if ref_xs:
            ax.plot(ref_xs, ref_ys, "s--", label="GraphCCS", color="tab:orange")

        ax.set_xlabel("Training fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Per-adduct table printer
# ---------------------------------------------------------------------------

def print_per_adduct(exp_dir: str, split_name: str, seeds):
    d = os.path.join(exp_dir, split_name)
    records = load_seed_metrics(d, seeds)
    if not records:
        return
    df = per_adduct_summary(records, "test")
    if df.empty:
        return
    print(f"\nPer-adduct (test) — {split_name}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments",
                        help="Root experiments directory")
    parser.add_argument("--out-dir", default="results",
                        help="Directory for tables and plots")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Main splits table ---
    print("\n" + "="*60)
    print("  Main splits summary (test set)")
    print("="*60)
    main_df = build_main_table(args.exp_dir, args.seeds)
    if not main_df.empty:
        print(main_df.to_string(index=False))
        main_df.to_csv(os.path.join(args.out_dir, "main_splits.csv"), index=False)

    # Per-adduct breakdown
    for split_name in MAIN_SPLITS:
        print_per_adduct(args.exp_dir, split_name, args.seeds)

    # --- Random learning curve (Exp D) ---
    print("\n" + "="*60)
    print("  Exp D — Random learning curve (test set)")
    print("="*60)
    lc_d = build_lc_table(args.exp_dir, "random_frac", args.seeds)
    if not lc_d.empty:
        print(lc_d.to_string(index=False))
        lc_d.to_csv(os.path.join(args.out_dir, "lc_random.csv"), index=False)

    lc_d_data = build_lc_data(args.exp_dir, "random_frac", args.seeds)
    plot_learning_curve(
        lc_d_data, GRAPHCCS_REF,
        os.path.join(args.out_dir, "lc_random.png"),
        title="Exp D — Random learning curve",
    )

    # --- Adduct-sensitive learning curve (Exp E) ---
    print("\n" + "="*60)
    print("  Exp E — Adduct-sensitive learning curve (test set)")
    print("="*60)
    lc_e = build_lc_table(args.exp_dir, "adduct_sensitive_frac", args.seeds)
    if not lc_e.empty:
        print(lc_e.to_string(index=False))
        lc_e.to_csv(os.path.join(args.out_dir, "lc_adduct_sensitive.csv"), index=False)

    lc_e_data = build_lc_data(args.exp_dir, "adduct_sensitive_frac", args.seeds)
    plot_learning_curve(
        lc_e_data, {},
        os.path.join(args.out_dir, "lc_adduct_sensitive.png"),
        title="Exp E — Adduct-sensitive learning curve",
    )

    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
