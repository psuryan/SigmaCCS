# SigmaCCS Experiments — Record of Work

## Goal

Retrain and evaluate SigmaCCS as a baseline model under the same experimental splits used for GraphCCS, reporting identical metrics so the two models can be compared directly.

Data and split definitions come from GraphCCS (sibling repo `../GraphCCS`). Data files are referenced by relative path and not duplicated here.

---

## Script Inventory

| Script | Purpose |
|--------|---------|
| `scripts/prepare_conformers.py` | Generate 3D conformers for all 9209 molecules in `data.csv` once. Saves `data/conformers.pkl`. Must be run before any training. |
| `scripts/run_experiment.py` | Train + evaluate for one split across multiple seeds. Args: `--data`, `--split`, `--conformers`, `--out`, `--seeds`, `--epochs`, `--batch-size`, `--label`. |
| `scripts/run_all.sh` | Shell wrapper that invokes `run_experiment.py` for all splits (Exps A–E) sequentially. |
| `scripts/prepare_split_csv.py` | Utility: given `data.csv` and `split.json`, write `train.csv`, `val.csv`, `test.csv` with SigmaCCS column names. |
| `scripts/run_analysis.py` | Aggregate `metrics.json` files across seeds; produce summary tables and learning curve plots. |

### Run order
```bash
# Step 1 — run once for all experiments
python scripts/prepare_conformers.py \
    --data ../GraphCCS/data/data.csv \
    --out  data/conformers.pkl

# Step 2 — per experiment
python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --split       ../GraphCCS/data/splits/random/split.json \
    --conformers  data/conformers.pkl \
    --out         experiments/random \
    --seeds       0 1 2 3 4 \
    --epochs      500 \
    --batch-size  32 \
    --label       random
```

---

## Data Layout

Shared with GraphCCS — referenced by absolute path, not copied.

```
../GraphCCS/data/
  data.csv                              — 9209 rows; columns: index, smiles, adducts, label
  splits/
    random/split.json                   — 7374 / 913 / 922  (train/val/test)
    scaffold/split.json                 — 7369 / 920 / 920
    adduct_sensitive/split.json         — 6446 / 1381 / 1382
    random_frac/
      split_0.1.json                    —  737 / 913 / 922
      split_0.2.json                    — 1474 / 913 / 922
      split_0.4.json                    — 2949 / 913 / 922
      split_0.6.json                    — 4424 / 913 / 922
      split_0.8.json                    — 5899 / 913 / 922
    adduct_sensitive_frac/
      split_0.1.json                    —  645 / 1381 / 1382
      split_0.2.json                    — 1289 / 1381 / 1382
      split_0.4.json                    — 2578 / 1381 / 1382
      split_0.6.json                    — 3868 / 1381 / 1382
      split_0.8.json                    — 5157 / 1381 / 1382
```

Conformer cache: `data/conformers.pkl` — 6258 unique SMILES, 0 failures, generated once.

---

## Model Configuration

| Parameter | Value | Paper value |
|-----------|-------|-------------|
| Optimizer | Adam | Adam |
| Learning rate | 0.0001 (fixed) | 0.0001 (fixed) |
| Batch size | 32 | 14 |
| Epochs | 500 | 300 |
| ECC layers | 3 × 16 channels | 3 × 16 channels |
| Kernel network | [64, 64, 64, 64] | [64, 64, 64, 64] |
| Dense layers | 8 × Dense(384, relu) + Dense(1, relu) | same |
| L2 regularization | l2=0.01 (all layers) | l2=0.01 (all layers) |
| Checkpoint | final epoch only | final epoch only |
| Best-val selection | not yet applied | not described |

**Notes:**
- Batch size increased to 32 (from paper's 14) — fits in 11 GB GPU (10.8 GB used), produces smoother gradients and faster per-epoch convergence.
- 500 epochs used because loss was still declining at epoch 300 on our larger training set (7374 vs paper's 5038 molecules).
- Best-val checkpoint selection not yet implemented — results below use the final-epoch model. This is a known limitation; see Known Issues.

---

## Experiment 1 — Random Split (5 Seeds)

**Split**: `random/split.json` — 7374 / 913 / 922 (train/val/test)
**Seeds**: 0–4
**Epochs**: 500 | **Batch size**: 32
**Output**: `experiments/random/`

### Command
```bash
python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --split       ../GraphCCS/data/splits/random/split.json \
    --conformers  data/conformers.pkl \
    --out         experiments/random \
    --seeds       0 1 2 3 4 \
    --epochs      500 \
    --batch-size  32 \
    --label       random
```

### Training loss curve (seed 0, representative)

| Epoch | Loss (MSE + L2) | sqrt(loss) |
|-------|-----------------|------------|
| 1 | 17999 | 134.2 |
| 51 | 68 | 8.3 |
| 101 | 39 | 6.2 |
| 201 | 31 | 5.5 |
| 301 | 22 | 4.7 |
| 401 | 17 | 4.2 |
| 500 | ~15 | ~3.9 |

Loss still declining at epoch 300 (justifying 500 epochs). Reported loss includes L2 regularization penalty — actual MSE component is lower.

### Per-seed test results

| Seed | RMSE | Mean%Diff | Pearson R | Spearman R | Kendall τ |
|------|------|-----------|-----------|------------|-----------|
| 0 | 5.1060 | 1.9382% | 0.995549 | 0.992948 | 0.933493 |
| 1 | 5.4066 | 2.1758% | 0.995377 | 0.992666 | 0.931868 |
| 2 | 5.1206 | 2.0241% | 0.996048 | 0.993727 | 0.937856 |
| 3 | 4.9274 | 1.8832% | 0.995906 | 0.993754 | 0.937399 |
| 4 | 5.3424 | 1.9997% | 0.995400 | 0.993143 | 0.933917 |

### Summary results (mean ± std across 5 seeds, **final-epoch model**)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| RMSE (Å²) | 3.525 ± 0.465 | 4.809 ± 0.214 | **5.181 ± 0.174** |
| Mean%Diff | 1.399% ± 0.224% | 1.908% ± 0.141% | **2.004% ± 0.099%** |
| Pearson R | 0.9980 ± 0.0006 | 0.9963 ± 0.0003 | **0.9957 ± 0.0003** |
| Spearman R | 0.9972 ± 0.0009 | 0.9950 ± 0.0003 | **0.9932 ± 0.0004** |
| Kendall τ | 0.9576 ± 0.0067 | 0.9419 ± 0.0018 | **0.9349 ± 0.0023** |

### Per-adduct results (mean ± std, test set)

| Adduct | n | RMSE | Mean%Diff |
|--------|---|------|-----------|
| [M+H]+ | 432 | 5.119 ± 0.280 | 2.066% ± 0.183% |
| [M+Na]+ | 229 | 5.336 ± 0.118 | 2.050% ± 0.080% |
| [M-H]- | 261 | 5.136 ± 0.207 | 1.863% ± 0.051% |

### Comparison with GraphCCS (random split, best-val model)

| Metric | SigmaCCS (final epoch) | GraphCCS (best-val) | Δ |
|--------|------------------------|---------------------|---|
| Test RMSE | 5.181 ± 0.174 | 4.791 ± 0.069 | +0.390 |
| Test Mean%Diff | 2.004% ± 0.099% | 1.729% ± 0.038% | +0.275% |
| Test Pearson R | 0.9957 ± 0.0003 | 0.9961 ± 0.0001 | −0.0004 |
| Test Spearman R | 0.9932 ± 0.0004 | 0.9939 ± 0.0004 | −0.0007 |
| Test Kendall τ | 0.9349 ± 0.0023 | 0.9405 ± 0.0017 | −0.0056 |

**Note**: SigmaCCS results use the final-epoch model; GraphCCS uses best-val checkpoint selection. The gap is expected to narrow once best-val selection is implemented for SigmaCCS.

### Generalization gap

Train–test RMSE gap: **3.525 → 5.181 = 1.656 Å²**. Large gap relative to GraphCCS (1.831 Å² at epoch 200) suggests final-epoch overfitting — best-val model should recover some test performance.

---

## Known Issues / Next Steps

1. **No best-val checkpoint selection** — current results use the final-epoch model. Plan: monkey-patch `tf.keras.Model.fit` (called once per epoch inside `sigma/model.py`) to save checkpoints every 50 epochs and always at epoch 300 (paper setting). After training, evaluate each checkpoint on val and select the best. This matches the GraphCCS methodology.

2. **No early stopping** — all seeds run the full 500 epochs regardless of val performance. Exception-based early stopping via the same `Model.fit` patch is possible; evaluate after best-val results are in hand.

3. **Epoch 300 metrics unknown for seeds 0–1** — training was completed before checkpointing was implemented. Seeds 0–1 will need to be rerun to get epoch-300 reference metrics for paper comparison.

4. **Seed variance is high** — test RMSE range 4.93–5.41 across 5 seeds (Δ=0.48 Å²). Seed 1 is an outlier (train RMSE 4.32 vs ~3.3 for other seeds), suggesting it hit a worse local minimum. Best-val selection will not fully address this; the stochastic weight init and training shuffle are the source.

5. **Conformer randomness** — `ps.randomSeed = -1` in `Generating_coordinates` means conformers are stochastic at generation time. The global cache (`data/conformers.pkl`) fixes conformers across all experiments, removing this source of variance. This is scientifically correct (conformers are a property of the molecule, not the split) but differs from the paper's original setup where conformers were regenerated each run.
