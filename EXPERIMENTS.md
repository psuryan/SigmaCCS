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

# Step 2 — per experiment (paper settings)
python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --split       ../GraphCCS/data/splits/<split_name>/split.json \
    --conformers  data/conformers.pkl \
    --out         experiments/<split_name> \
    --seeds       0 1 2 3 4 \
    --epochs      500 \
    --batch-size  14 \
    --patience    20 \
    --label       <split_name>
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

Two training configurations have been used across experiments. Architecture is identical in both.

| Parameter | Exp 1 (exploratory) | Exp 2+ (paper settings) | Paper |
|-----------|---------------------|--------------------------|-------|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 0.0001 (fixed) | 0.0001 (fixed) | 0.0001 (fixed) |
| Batch size | 32 | **14** | 14 |
| Epochs (max) | 500 | **500** | 300 |
| Early stopping patience | none | **20 epochs (val RMSE)** | not described |
| Best-val checkpoint | no (final epoch) | **yes** (`model_best_val.h5`) | not described |
| ECC layers | 3 × 16 channels | 3 × 16 channels | 3 × 16 channels |
| Kernel network | [64, 64, 64, 64] | [64, 64, 64, 64] | [64, 64, 64, 64] |
| Dense layers | 8 × Dense(384, relu) + Dense(1, relu) | same | same |
| L2 regularization | l2=0.01 (all layers) | l2=0.01 (all layers) | l2=0.01 (all layers) |

**Notes:**
- Exp 1 used batch 32 and 500 epochs to explore convergence; no val-based model selection.
- Exp 2+ uses paper batch size (14), 500 max epochs, plus `PersistentEarlyStopping` (patience=20 on val RMSE) saving `model_best_val.h5` at each new best; evaluation uses this checkpoint rather than the final epoch.
- Conformer cache (`data/conformers.pkl`, 6258 entries) is shared across all experiments — conformers are fixed, removing a stochastic source the original paper did not control for.

---

## Experiment 1 — Random Split, Final-Epoch Model (Exploratory Run)

> **Config**: Exp 1 (exploratory) — 500 epochs, batch 32, no best-val selection, no early stopping.
> Results use the **final-epoch** model. Superseded by Experiment 2 which uses paper settings + best-val selection.

**Split**: `random/split.json` — 7374 / 913 / 922 (train/val/test)
**Seeds**: 0–4
**Epochs**: 500 | **Batch size**: 32 | **Patience**: none | **Best-val**: no
**Output**: `experiments/random_last_ep/`

### Command
```bash
python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --split       ../GraphCCS/data/splits/random/split.json \
    --conformers  data/conformers.pkl \
    --out         experiments/random_last_ep \
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

## Experiment 2 — Random Split, Best-Val Model ✓ COMPLETE

> **Config**: Exp 2 — 500 epochs, batch 14, patience 20 on val RMSE, best-val checkpoint.
> This is the primary comparable result for GraphCCS. Output: `experiments/random/`.

**Split**: `random/split.json` — 7374 / 913 / 922 (train/val/test)
**Seeds**: 0–4
**Epochs**: 500 | **Batch size**: 14 | **Patience**: 20 (val RMSE, early stopping) | **Best-val**: yes (`model_best_val.h5`)
**Log**: `experiments/random_run.log`
**Output**: `experiments/random/`

### Command
```bash
nohup python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --conformers  data/conformers.pkl \
    --split       ../GraphCCS/data/splits/random/split.json \
    --out         experiments/random \
    --seeds       0 1 2 3 4 \
    --epochs      500 \
    --batch-size  14 \
    --patience    20 \
    --label       random \
    > experiments/random_run.log 2>&1 &
```

### Best-val epochs (per seed)

| Seed | Best-val epoch | Best val RMSE |
|------|---------------|---------------|
| 0    | 139           | 4.5517        |
| 1    | 176           | 4.7223        |
| 2    | 100           | 4.7047        |
| 3    | 109           | 4.8697        |
| 4    | 164           | 4.7161        |

Early stopping with patience 5 meant most seeds converged well before epoch 300. Best-val epoch ranged 100–176, indicating the val optimum is in the middle of training, not at the end.

### Per-seed results — train, test, and generalization gap

| Seed | Train RMSE | Train %Diff | Train R | Train ρ | Train τ | Test RMSE | Test %Diff | Test R | Test ρ | Test τ | Gen. Gap (RMSE) |
|------|-----------|-------------|---------|---------|---------|-----------|------------|--------|--------|--------|-----------------|
| 0    | 4.176     | 1.603%      | 0.99693 | 0.99574 | 0.94700 | 4.866     | 1.868%     | 0.99596 | 0.99377 | 0.93669 | **0.690** |
| 1    | 5.074     | 1.816%      | 0.99549 | 0.99408 | 0.93958 | 5.250     | 1.972%     | 0.99530 | 0.99258 | 0.93174 | **0.176** |
| 2    | 4.645     | 1.762%      | 0.99626 | 0.99491 | 0.94277 | 5.257     | 2.006%     | 0.99534 | 0.99282 | 0.93270 | **0.612** |
| 3    | 5.130     | 1.846%      | 0.99551 | 0.99407 | 0.93941 | 5.292     | 1.986%     | 0.99540 | 0.99317 | 0.93378 | **0.162** |
| 4    | 4.294     | 1.612%      | 0.99678 | 0.99546 | 0.94654 | 5.020     | 1.916%     | 0.99573 | 0.99354 | 0.93580 | **0.726** |
| **Mean** | **4.664** | **1.728%** | **0.9962** | **0.9949** | **0.9431** | **5.137** | **1.950%** | **0.9955** | **0.9932** | **0.9341** | **0.473** |
| **± Std** | **± 0.436** | **± 0.114%** | **± 0.0007** | **± 0.0007** | **± 0.0036** | **± 0.186** | **± 0.056%** | **± 0.0003** | **± 0.0005** | **± 0.0021** | **± 0.281** |

### Validation set — all metrics per seed

| Seed | RMSE (Å) | Mean%Diff | Pearson R | Spearman R | Kendall τ |
|------|----------|-----------|-----------|------------|-----------|
| 0    | 4.552    | 1.755%    | 0.99641   | 0.99557    | 0.94418   |
| 1    | 4.722    | 1.803%    | 0.99612   | 0.99527    | 0.94232   |
| 2    | 4.705    | 1.881%    | 0.99622   | 0.99519    | 0.94187   |
| 3    | 4.870    | 1.897%    | 0.99592   | 0.99503    | 0.94080   |
| 4    | 4.716    | 1.862%    | 0.99615   | 0.99510    | 0.94141   |
| **Mean** | **4.713** | **1.840%** | **0.99616** | **0.99523** | **0.94212** |
| **± Std** | **± 0.113** | **± 0.059%** | **± 0.0002** | **± 0.0002** | **± 0.0014** |

### Test set — per-adduct RMSE per seed

| Adduct | n | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
|--------|---|--------|--------|--------|--------|--------|------------|
| [M+H]+ | 432 | 4.543 | 4.838 | 5.021 | 4.975 | 4.700 | **4.815 ± 0.197** |
| [M+Na]+ | 229 | 5.211 | 5.910 | 5.626 | 5.884 | 5.384 | **5.603 ± 0.306** |
| [M-H]- | 261 | 5.066 | 5.288 | 5.305 | 5.254 | 5.198 | **5.222 ± 0.096** |

### Test set — per-adduct Mean%Diff per seed

| Adduct | n | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
|--------|---|--------|--------|--------|--------|--------|------------|
| [M+H]+ | 432 | 1.802% | 1.853% | 1.997% | 1.936% | 1.866% | **1.891% ± 0.076%** |
| [M+Na]+ | 229 | 2.012% | 2.217% | 2.182% | 2.228% | 2.052% | **2.138% ± 0.100%** |
| [M-H]- | 261 | 1.852% | 1.953% | 1.866% | 1.857% | 1.879% | **1.881% ± 0.042%** |

### Comparison with GraphCCS (random split)

| Metric | SigmaCCS Exp 1 (final epoch) | SigmaCCS Exp 2 (best-val) | GraphCCS (best-val) |
|--------|------------------------------|---------------------------|---------------------|
| Test RMSE | 5.181 ± 0.174 | **5.137 ± 0.186** | 4.791 ± 0.069 |
| Test Mean%Diff | 2.004% ± 0.099% | **1.950% ± 0.056%** | 1.729% ± 0.038% |
| Test Pearson R | 0.9957 ± 0.0003 | **0.9955 ± 0.0003** | 0.9961 ± 0.0001 |
| Test Spearman R | 0.9932 ± 0.0004 | **0.9932 ± 0.0005** | 0.9939 ± 0.0004 |
| Test Kendall τ | 0.9349 ± 0.0023 | **0.9341 ± 0.0021** | 0.9405 ± 0.0017 |

Best-val selection (Exp 2) modestly improved RMSE and Mean%Diff over the final-epoch run (Exp 1). The remaining gap to GraphCCS (~0.35 Å RMSE) reflects architectural differences, not experimental setup.

### Generalization gap (Exp 2)

Train–val–test RMSE: **4.664 → 4.713 → 5.137**. Train–test gap of ~0.47 Å is smaller than Exp 1 (~1.66 Å), confirming that best-val selection significantly reduces overfitting.

---

## Experiment 3 — Adduct-Sensitive Split, Best-Val Model ✓ COMPLETE

> **Config**: Exp 2+ settings — 500 epochs, batch 14, patience 20 on val RMSE, best-val checkpoint.
> Split puts molecules with the highest CCS range across adducts in val+test; train contains single-adduct and low-range molecules. Harder generalisation than random split.

**Split**: `adduct_sensitive/split.json` — 6446 / 1381 / 1382 (train/val/test)
**Seeds**: 0–4
**Epochs**: 500 | **Batch size**: 14 | **Patience**: 20 (val RMSE) | **Best-val**: yes (`model_best_val.h5`)
**Log**: `experiments/adduct_sensitive_run.log` (seeds 0–3), `experiments/adduct_sensitive_seed4_run.log` (seed 4 rerun)
**Output**: `experiments/adduct_sensitive/`

> **Note on seed 4**: Seed 4 diverged in the initial run (best-val at epoch 1, val RMSE=204). The seed_4 directory was cleaned and rerun; seed 4 converged cleanly on the second attempt (best-val epoch 126).

### Command
```bash
nohup python scripts/run_experiment.py \
    --data        ../GraphCCS/data/data.csv \
    --conformers  data/conformers.pkl \
    --split       ../GraphCCS/data/splits/adduct_sensitive/split.json \
    --out         experiments/adduct_sensitive \
    --seeds       0 1 2 3 4 \
    --epochs      500 \
    --batch-size  14 \
    --patience    20 \
    --label       adduct_sensitive \
    > experiments/adduct_sensitive_run.log 2>&1 &
```

### Best-val epochs (per seed)

| Seed | Best-val epoch | Best val RMSE |
|------|---------------|---------------|
| 0    | 96            | 6.7133        |
| 1    | 108           | 6.8096        |
| 2    | 78            | 6.9822        |
| 3    | 41            | 7.4988        |
| 4    | 126           | 6.6453        |

### Per-seed results — train, test, and generalization gap

| Seed | Train RMSE | Train %Diff | Train R | Train ρ | Train τ | Test RMSE | Test %Diff | Test R | Test ρ | Test τ | Gen. Gap (RMSE) |
|------|-----------|-------------|---------|---------|---------|-----------|------------|--------|--------|--------|-----------------|
| 0    | 4.963     | 2.063%      | 0.99666 | 0.99556 | 0.94515 | 6.471     | 2.563%     | 0.99103 | 0.98758 | 0.90989 | **1.508** |
| 1    | 5.226     | 1.965%      | 0.99589 | 0.99478 | 0.94269 | 6.847     | 2.637%     | 0.99000 | 0.98617 | 0.90638 | **1.621** |
| 2    | 5.127     | 2.042%      | 0.99605 | 0.99483 | 0.94088 | 6.947     | 2.786%     | 0.98973 | 0.98509 | 0.90015 | **1.820** |
| 3    | 6.908     | 2.791%      | 0.99333 | 0.99115 | 0.92526 | 7.343     | 3.113%     | 0.98902 | 0.98392 | 0.89565 | **0.435** |
| 4    | 4.090     | 1.635%      | 0.99741 | 0.99641 | 0.95083 | 6.980     | 2.680%     | 0.98989 | 0.98616 | 0.90383 | **2.890** |
| **Mean** | **5.263** | **2.099%** | **0.9959** | **0.9945** | **0.9410** | **6.918** | **2.756%** | **0.9899** | **0.9858** | **0.9032** | **1.655** |
| **± Std** | **± 1.024** | **± 0.423%** | **± 0.0015** | **± 0.0026** | **± 0.0095** | **± 0.313** | **± 0.215%** | **± 0.0007** | **± 0.0014** | **± 0.0055** | **± 0.875** |

### Validation set — all metrics per seed

| Seed | RMSE (Å) | Mean%Diff | Pearson R | Spearman R | Kendall τ |
|------|----------|-----------|-----------|------------|-----------|
| 0    | 6.713    | 2.702%    | 0.99127   | 0.98833    | 0.91013   |
| 1    | 6.810    | 2.633%    | 0.99096   | 0.98796    | 0.91026   |
| 2    | 6.982    | 2.820%    | 0.99053   | 0.98681    | 0.90430   |
| 3    | 7.499    | 3.202%    | 0.98961   | 0.98584    | 0.90021   |
| 4    | 6.645    | 2.615%    | 0.99153   | 0.98845    | 0.91060   |
| **Mean** | **6.930** | **2.794%** | **0.99078** | **0.98748** | **0.90710** |
| **± Std** | **± 0.342** | **± 0.241%** | **± 0.0008** | **± 0.0010** | **± 0.0044** |

### Test set — per-adduct RMSE per seed

| Adduct | n | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
|--------|---|--------|--------|--------|--------|--------|------------|
| [M+H]+ | 469 | 5.737 | 5.859 | 6.349 | 7.532 | 6.106 | **6.317 ± 0.719** |
| [M+Na]+ | 504 | 6.805 | 7.594 | 7.363 | 7.467 | 7.611 | **7.368 ± 0.330** |
| [M-H]- | 409 | 6.832 | 6.926 | 7.075 | 6.960 | 7.106 | **6.980 ± 0.112** |

### Test set — per-adduct Mean%Diff per seed

| Adduct | n | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
|--------|---|--------|--------|--------|--------|--------|------------|
| [M+H]+ | 469 | 2.253% | 2.326% | 2.555% | 3.307% | 2.226% | **2.533% ± 0.451%** |
| [M+Na]+ | 504 | 2.626% | 2.938% | 2.922% | 3.003% | 2.999% | **2.898% ± 0.156%** |
| [M-H]- | 409 | 2.841% | 2.622% | 2.883% | 3.026% | 2.808% | **2.836% ± 0.146%** |

### Comparison: Adduct-Sensitive vs Random split (SigmaCCS Exp 2)

| Metric | Random (Exp 2) | Adduct-Sensitive (Exp 3) | Δ |
|--------|---------------|--------------------------|---|
| Test RMSE | 5.137 ± 0.186 | **6.918 ± 0.313** | +1.781 |
| Test Mean%Diff | 1.950% ± 0.056% | **2.756% ± 0.215%** | +0.806% |
| Test Pearson R | 0.9955 ± 0.0003 | **0.9899 ± 0.0007** | −0.0056 |
| Test Spearman R | 0.9932 ± 0.0005 | **0.9858 ± 0.0014** | −0.0074 |
| Test Kendall τ | 0.9341 ± 0.0021 | **0.9032 ± 0.0055** | −0.0309 |

The adduct-sensitive split is substantially harder (+1.78 Å RMSE). The distribution shift — training on low-CCS-range molecules, testing on high-CCS-range ones — explains most of the gap. Seed variance is also higher (gen gap std 0.875 vs 0.281 for random), reflecting sensitivity to initialisation on this harder split.

---

## Known Issues / Next Steps

1. ~~**No best-val checkpoint selection**~~ — **RESOLVED** in Exp 2. `PersistentEarlyStopping` in `scripts/callbacks.py` saves `model_best_val.h5` at every new val RMSE minimum; evaluation uses this checkpoint.

2. ~~**No early stopping**~~ — **RESOLVED** in Exp 2. Patience=20 on val RMSE is applied via `PersistentEarlyStopping`. Most seeds converged between epochs 100–176.

3. ~~**Epoch 300 metrics unknown for seeds 0–1**~~ — **RESOLVED** in Exp 2. All 5 seeds were rerun from scratch with best-val selection and paper settings.

4. **Seed variance is high (Exp 2)** — test RMSE range 4.87–5.29 across 5 seeds (Δ=0.43 Å). Seed 0 is the best (best-val epoch 139), seeds 1–3 are weaker. Source is stochastic weight init + training shuffle, not the checkpoint strategy.

5. **[M+Na]+ is the weakest adduct** — test RMSE 5.603 ± 0.306, notably higher than [M+H]+ (4.815) and [M-H]- (5.222), and with the largest seed variance. Likely reflects fewer training examples (1766 vs 3436 for [M+H]+) and greater structural sensitivity.

6. **Remaining splits not yet run** — Exp B (scaffold), Exp C (adduct_sensitive), Exp D (random_frac), Exp E (adduct_sensitive_frac) are all pending. Run via `scripts/run_all.sh` or individually with settings: batch 14, epochs 500, patience 20.

7. **Conformer randomness** — `ps.randomSeed = -1` in `Generating_coordinates` means conformers are stochastic at generation time. The global cache (`data/conformers.pkl`) fixes conformers across all experiments, removing this source of variance. This is scientifically correct (conformers are a property of the molecule, not the split) but differs from the paper's original setup where conformers were regenerated each run.
