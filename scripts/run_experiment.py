"""
Main training + evaluation script for SigmaCCS baseline experiments.

Requires a pre-built conformer cache (run prepare_conformers.py first):
    python scripts/prepare_conformers.py \
        --data ../GraphCCS/data/data.csv \
        --out  data/conformers.pkl

Example:
    python scripts/run_experiment.py \
        --data        ../GraphCCS/data/data.csv \
        --split       ../GraphCCS/data/splits/random/split.json \
        --conformers  data/conformers.pkl \
        --out         experiments/random \
        --seeds       0 1 2 3 4 \
        --epochs      300 \
        --batch-size  14 \
        --label       random
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import tensorflow as tf

import sigma.GraphData as _gd
import sigma.sigma as _ss
from sigma.sigma import Model_train, Model_prediction
from scripts.callbacks import PersistentEarlyStopping, EarlyStopSignal
from scripts.prepare_split_csv import prepare_split_csvs


# ---------------------------------------------------------------------------
# Conformer cache patch
# ---------------------------------------------------------------------------

def _load_conformer_cache(pkl_path: str) -> dict:
    """Load the global conformer cache produced by prepare_conformers.py."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return payload["conformers"]   # dict: smiles -> coords (or None)


def _make_cached_fn(conformer_cache: dict):
    """
    Return a drop-in replacement for sigma.GraphData.Generating_coordinates
    that uses pre-computed conformers instead of re-running ETKDGv3+MMFF94.

    The replacement replicates all filtering from the original:
      - MolFromSmiles failure
      - single-atom / no-bond molecules
      - elements not in All_Atoms
      - conformer generation failure (cache value is None)
    so downstream code (Model_train, Model_prediction) sees exactly the same
    filtered molecule set it would have seen had it run from scratch.
    """
    def cached_generating_coordinates(smiles, adduct, ccs, All_Atoms, ps=None):
        succ_smiles, succ_adduct, succ_ccs, coordinates = [], [], [], []
        All_Atoms_set = set(All_Atoms)
        for i, smi in enumerate(smiles):
            # --- replicate the element / structure filters ---
            try:
                iMol = Chem.MolFromSmiles(smi)
                if iMol is None:
                    continue
                iMol = Chem.RemoveHs(iMol)
                atoms = [atom.GetSymbol() for atom in iMol.GetAtoms()]
                bonds = list(iMol.GetBonds())
            except Exception:
                continue
            if len(atoms) == 1 and len(bonds) <= 1:
                continue
            if any(a not in All_Atoms_set for a in atoms):
                continue
            # --- use cached coordinates ---
            coords = conformer_cache.get(smi)
            if coords is None:
                continue
            succ_smiles.append(smi)
            succ_adduct.append(adduct[i])
            succ_ccs.append(ccs[i])
            coordinates.append(coords)
        return succ_smiles, succ_adduct, succ_ccs, coordinates

    return cached_generating_coordinates


def _patch(cached_fn):
    original = _gd.Generating_coordinates
    _gd.Generating_coordinates = cached_fn
    _ss.Generating_coordinates = cached_fn
    return original


def _unpatch(original):
    _gd.Generating_coordinates = original
    _ss.Generating_coordinates = original


# ---------------------------------------------------------------------------
# Model.fit patch — injects persistent callbacks into sigma's training loop
# ---------------------------------------------------------------------------

def _patch_model_fit(callbacks):
    """
    Monkey-patch tf.keras.Model.fit to append *callbacks* to every call.

    sigma/model.py calls Model.fit(..., epochs=1) once per epoch in a loop.
    Standard Keras callbacks reset their state each call; the PersistentXxx
    callbacks in scripts/callbacks.py are designed to survive this.
    """
    original_fit = tf.keras.Model.fit

    def _patched_fit(self, *args, **kwargs):
        existing = list(kwargs.get("callbacks") or [])
        kwargs["callbacks"] = existing + callbacks
        return original_fit(self, *args, **kwargs)

    tf.keras.Model.fit = _patched_fit
    return original_fit


def _unpatch_model_fit(original_fit):
    tf.keras.Model.fit = original_fit


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(pred_csv: str) -> dict:
    df = pd.read_csv(pred_csv)
    true_col = "Ture CCS" if "Ture CCS" in df.columns else "True CCS"
    true = df[true_col].values.astype(float)
    pred = df["Predicted CCS"].values.astype(float)

    def _metrics(t, p):
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mean_pct_diff = float(np.mean(np.abs(p - t) / t) * 100)
        pearson_r, _ = stats.pearsonr(t, p)
        spearman_r, _ = stats.spearmanr(t, p)
        kendall_tau, _ = stats.kendalltau(t, p)
        return {
            "rmse": round(rmse, 6),
            "mean_pct_diff": round(mean_pct_diff, 6),
            "pearson_r": round(float(pearson_r), 6),
            "spearman_r": round(float(spearman_r), 6),
            "kendall_tau": round(float(kendall_tau), 6),
        }

    overall = _metrics(true, pred)

    per_adduct = {}
    for adduct, grp in df.groupby("Adduct"):
        t = grp[true_col].values.astype(float)
        p = grp["Predicted CCS"].values.astype(float)
        per_adduct[adduct] = {
            "rmse": round(float(np.sqrt(np.mean((p - t) ** 2))), 6),
            "mean_pct_diff": round(float(np.mean(np.abs(p - t) / t) * 100), 6),
            "n": len(t),
        }

    overall["per_adduct"] = per_adduct
    return overall


# ---------------------------------------------------------------------------
# Single-seed experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int, train_csv: str, val_csv: str, test_csv: str,
             out_dir: str, epochs: int, batch_size: int, cached_fn,
             conformer_cache: dict, patience: int = 10):
    os.makedirs(out_dir, exist_ok=True)

    final_model_path = os.path.join(out_dir, "model_final.h5")
    param_path      = os.path.join(out_dir, "parameter.pkl")
    train_pred_path = os.path.join(out_dir, "train.csv")
    val_pred_path   = os.path.join(out_dir, "val.csv")
    test_pred_path  = os.path.join(out_dir, "test.csv")
    metrics_path    = os.path.join(out_dir, "metrics.json")

    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  Seed {seed}  ->  {out_dir}")
    print(f"{'='*60}")

    # --- Callbacks ---
    es_cb = PersistentEarlyStopping(
        val_csv=val_csv,
        param_path=param_path,
        conformer_cache=conformer_cache,
        patience=patience,
    )

    # --- Train ---
    orig_conformer = _patch(cached_fn)
    orig_fit = _patch_model_fit([es_cb])
    try:
        print("\n[TRAIN]")
        Model_train(
            ifile=train_csv,
            ParameterPath=param_path,
            ofile=final_model_path,
            EPOCHS=epochs,
            BATCHS=batch_size,
            Vis=1,
            All_Atoms=[],
            adduct_SET=[],
        )
    except EarlyStopSignal as e:
        print(f"\n[EARLY STOP] {e}")
    finally:
        _unpatch(orig_conformer)
        _unpatch_model_fit(orig_fit)

    # --- Select best-val model ---
    # PersistentEarlyStopping saves model_best_val.h5 whenever val RMSE improves.
    # Fall back to the final-epoch model if no val check ran (shouldn't happen).
    best_model_path = es_cb.best_model_path or final_model_path
    best_val_epoch  = es_cb.best_epoch
    best_val_rmse   = es_cb.best_val_rmse if es_cb.best_val_rmse < np.inf else None
    print(f"\n[BEST-VAL] Using epoch={best_val_epoch}  val_rmse={best_val_rmse}  -> {os.path.basename(best_model_path)}")

    # --- Predict train / val / test using best-val model ---
    for split_name, csv_path, pred_path in [
        ("TRAIN", train_csv, train_pred_path),
        ("VAL",   val_csv,   val_pred_path),
        ("TEST",  test_csv,  test_pred_path),
    ]:
        orig = _patch(cached_fn)
        try:
            print(f"\n[PREDICT {split_name}]")
            Model_prediction(
                ifile=csv_path,
                ParameterPath=param_path,
                mfileh5=best_model_path,
                ofile=pred_path,
                Isevaluate=1,
            )
        finally:
            _unpatch(orig)

    metrics = {
        "best_val_epoch": best_val_epoch,
        "best_val_rmse":  round(best_val_rmse, 6) if best_val_rmse is not None else None,
        "train": compute_metrics(train_pred_path),
        "val":   compute_metrics(val_pred_path),
        "test":  compute_metrics(test_pred_path),
    }
    for split_name in ("train", "val", "test"):
        m = metrics[split_name]
        print(f"  [{split_name.upper():5s}] RMSE={m['rmse']:.4f}  Mean%Diff={m['mean_pct_diff']:.4f}%  PearsonR={m['pearson_r']:.6f}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] metrics -> {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SigmaCCS experiment across seeds")
    parser.add_argument("--data",       required=True, help="Path to data.csv")
    parser.add_argument("--split",      required=True, help="Path to split.json")
    parser.add_argument("--conformers", required=True, help="Path to conformers.pkl (from prepare_conformers.py)")
    parser.add_argument("--out",        required=True, help="Output directory")
    parser.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs",     type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--patience",   type=int, default=20,
                        help="Early-stopping patience (number of val checks without improvement)")
    parser.add_argument("--label",      default="")
    args = parser.parse_args()

    label_str = f" [{args.label}]" if args.label else ""
    print(f"\nExperiment{label_str}")
    print(f"  data       : {args.data}")
    print(f"  split      : {args.split}")
    print(f"  conformers : {args.conformers}")
    print(f"  out        : {args.out}")
    print(f"  seeds      : {args.seeds}")
    print(f"  epochs     : {args.epochs}  batch: {args.batch_size}  patience: {args.patience}")

    # Build split CSVs
    csv_dir = os.path.join(args.out, "split_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    print("\nPreparing split CSVs...")
    train_csv, val_csv, test_csv = prepare_split_csvs(args.data, args.split, csv_dir)

    # Load global conformer cache — generated once for all 9209 molecules
    print(f"\nLoading conformer cache from {args.conformers}...")
    conformer_cache = _load_conformer_cache(args.conformers)
    print(f"  {len(conformer_cache)} entries loaded")
    cached_fn = _make_cached_fn(conformer_cache)

    all_metrics = {}
    for seed in args.seeds:
        seed_dir = os.path.join(args.out, f"seed_{seed}")
        seed_metrics = run_seed(
            seed=seed,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            out_dir=seed_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cached_fn=cached_fn,
            conformer_cache=conformer_cache,
            patience=args.patience,
        )
        all_metrics[f"seed_{seed}"] = seed_metrics

    print(f"\n{'='*60}")
    print(f"  Summary{label_str}  (test set)")
    print(f"{'='*60}")
    for metric in ("rmse", "mean_pct_diff", "pearson_r"):
        vals = [all_metrics[f"seed_{s}"]["test"][metric] for s in args.seeds]
        print(f"  {metric:>14s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
