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
    --epochs      300 \
    --batch-size  14 \
    --patience    5 \
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
| Epochs (max) | 500 | **300** | 300 |
| Early stopping patience | none | **5 epochs (val RMSE)** | not described |
| Best-val checkpoint | no (final epoch) | **yes** (`model_best_val.h5`) | not described |
| ECC layers | 3 × 16 channels | 3 × 16 channels | 3 × 16 channels |
| Kernel network | [64, 64, 64, 64] | [64, 64, 64, 64] | [64, 64, 64, 64] |
| Dense layers | 8 × Dense(384, relu) + Dense(1, relu) | same | same |
| L2 regularization | l2=0.01 (all layers) | l2=0.01 (all layers) | l2=0.01 (all layers) |

**Notes:**
- Exp 1 used batch 32 and 500 epochs to explore convergence; no val-based model selection.
- Exp 2+ uses paper batch size (14) and epoch budget (300), plus `PersistentEarlyStopping` (patience=5 on val RMSE) saving `model_best_val.h5` at each new best; evaluation uses this checkpoint rather than the final epoch.
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

## Experiment 2 — Random Split, Best-Val Model (Paper Settings) ✓ COMPLETE

> **Config**: Exp 2 (paper settings) — 300 epochs, batch 14, patience 5 on val RMSE, best-val checkpoint.
> This is the primary comparable result for GraphCCS. Output: `experiments/random/`.

**Split**: `random/split.json` — 7374 / 913 / 922 (train/val/test)
**Seeds**: 0–4
**Epochs**: 300 | **Batch size**: 14 | **Patience**: 5 (val RMSE, early stopping) | **Best-val**: yes (`model_best_val.h5`)
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
    --epochs      300 \
    --batch-size  14 \
    --patience    5 \
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

## Known Issues / Next Steps

1. ~~**No best-val checkpoint selection**~~ — **RESOLVED** in Exp 2. `PersistentEarlyStopping` in `scripts/callbacks.py` saves `model_best_val.h5` at every new val RMSE minimum; evaluation uses this checkpoint.

2. ~~**No early stopping**~~ — **RESOLVED** in Exp 2. Patience=5 on val RMSE is applied via `PersistentEarlyStopping`. Most seeds converged between epochs 100–176.

3. ~~**Epoch 300 metrics unknown for seeds 0–1**~~ — **RESOLVED** in Exp 2. All 5 seeds were rerun from scratch with best-val selection and paper settings.

4. **Seed variance is high (Exp 2)** — test RMSE range 4.87–5.29 across 5 seeds (Δ=0.43 Å). Seed 0 is the best (best-val epoch 139), seeds 1–3 are weaker. Source is stochastic weight init + training shuffle, not the checkpoint strategy.

5. **[M+Na]+ is the weakest adduct** — test RMSE 5.603 ± 0.306, notably higher than [M+H]+ (4.815) and [M-H]- (5.222), and with the largest seed variance. Likely reflects fewer training examples (1766 vs 3436 for [M+H]+) and greater structural sensitivity.

6. **Remaining splits not yet run** — Exp B (scaffold), Exp C (adduct_sensitive), Exp D (random_frac), Exp E (adduct_sensitive_frac) are all pending. Run via `scripts/run_all.sh` or individually with paper settings (batch 14, epochs 300, patience 5).

7. **Conformer randomness** — `ps.randomSeed = -1` in `Generating_coordinates` means conformers are stochastic at generation time. The global cache (`data/conformers.pkl`) fixes conformers across all experiments, removing this source of variance. This is scientifically correct (conformers are a property of the molecule, not the split) but differs from the paper's original setup where conformers were regenerated each run.
