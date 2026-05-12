"""
Persistent Keras callbacks for use with sigma/model.py's training loop.

sigma/model.py's train() calls Model.fit(..., epochs=1) once per epoch in a loop.
Standard Keras callbacks (ModelCheckpoint, EarlyStopping) call on_train_begin at
the start of each fit() call, resetting their internal state every epoch.
These callbacks override on_train_begin to initialise only once, accumulating
state correctly across all epochs.

Usage in run_experiment.py:
    from scripts.callbacks import PersistentEarlyStopping, EarlyStopSignal

    es_cb = PersistentEarlyStopping(
                val_csv=val_csv,
                param_path=param_path,
                conformer_cache=conformer_cache,
            )
    orig_fit = _patch_model_fit([es_cb])
    try:
        Model_train(...)
    except EarlyStopSignal as e:
        print(f'[EARLY STOP] {e}')
    finally:
        _unpatch_model_fit(orig_fit)

    # Best-val model path:
    best_path = es_cb.best_model_path   # <out_dir>/model_best_val.h5
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import rdkit.Chem as Chem


class EarlyStopSignal(Exception):
    """Raised by PersistentEarlyStopping to break out of sigma's training loop."""
    pass


class PersistentEarlyStopping(tf.keras.callbacks.Callback):
    """
    Early stopping on val RMSE. Saves model_best_val.h5 whenever val RMSE improves.

    Val RMSE is computed every check_val_every_n_epoch epochs by running direct
    model inference on the pre-built val graph dataset. The val dataset is
    constructed lazily on the first check using the conformer cache and
    parameter.pkl (written by Model_train before the training loop begins).

    Parameters
    ----------
    val_csv : str
        Path to val.csv (columns: SMILES, Adduct, True CCS).
    param_path : str
        Path to parameter.pkl written by Model_train.
    conformer_cache : dict
        Global conformer cache {smiles -> coordinate list} from prepare_conformers.py.
    patience : int
        Number of val checks (not epochs) without improvement before stopping.
        Default 20.
    min_delta : float
        Minimum RMSE improvement to count as an improvement.
    check_val_every_n_epoch : int
        Evaluate val RMSE every this many epochs. Default 1.
    """

    def __init__(
        self,
        val_csv: str,
        param_path: str,
        conformer_cache: dict,
        patience: int = 20,
        min_delta: float = 0.01,
        check_val_every_n_epoch: int = 1,
    ):
        super().__init__()
        self.val_csv = val_csv
        self.param_path = param_path
        self.conformer_cache = conformer_cache
        self.patience = patience
        self.min_delta = min_delta
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self._best_model_path = os.path.join(os.path.dirname(param_path), "model_best_val.h5")
        self._best_val_rmse = np.inf
        self._best_epoch = None

        self._epoch_count = 0
        self._initialized = False
        self._val_dataset = None   # built lazily
        self._val_adduct = None
        self._val_ccs = None
        self._adduct_set = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def best_model_path(self):
        return self._best_model_path if self._best_epoch is not None else None

    @property
    def best_val_rmse(self):
        return self._best_val_rmse

    @property
    def best_epoch(self):
        return self._best_epoch

    # ------------------------------------------------------------------
    # State persistence across fit(epochs=1) calls
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        if not self._initialized:
            self.wait = 0
            self._initialized = True
        # Do NOT reset on subsequent calls.

    # ------------------------------------------------------------------
    # Val dataset construction (lazy, on first check)
    # ------------------------------------------------------------------

    def _build_val_dataset(self):
        """
        Build the val graph dataset once from parameter.pkl + conformer cache.
        parameter.pkl is written by Model_train before the training loop begins,
        so it is always available when the first on_epoch_end fires.
        """
        REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from sigma.GraphData import convertToGraph, MyDataset

        with open(self.param_path, "rb") as f:
            param = pickle.load(f)

        All_Atoms     = param.All_Atoms
        All_Atoms_set = set(All_Atoms)
        Min_Coor      = param.Min_Coor
        Max_Coor      = param.Max_Coor
        adduct_SET    = param.adduct_SET

        df = pd.read_csv(self.val_csv)
        true_col = "Ture CCS" if "Ture CCS" in df.columns else "True CCS"
        smiles_list = df["SMILES"].tolist()
        adduct_list = df["Adduct"].tolist()
        ccs_list    = df[true_col].tolist()

        succ_smiles, succ_adduct, succ_ccs, coordinates = [], [], [], []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mol = Chem.RemoveHs(mol)
                atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                bonds = list(mol.GetBonds())
            except Exception:
                continue
            if len(atoms) == 1 and len(bonds) <= 1:
                continue
            if any(a not in All_Atoms_set for a in atoms):
                continue
            coords = self.conformer_cache.get(smi)
            if coords is None:
                continue
            norm_coords = [(np.array(c) - Min_Coor) / (Max_Coor - Min_Coor) for c in coords]
            succ_smiles.append(smi)
            succ_adduct.append(adduct_list[i])
            succ_ccs.append(ccs_list[i])
            coordinates.append(norm_coords)

        adj, features, edge_features = convertToGraph(succ_smiles, coordinates, All_Atoms)
        self._val_dataset = MyDataset(features, adj, edge_features, succ_ccs)
        self._val_adduct  = succ_adduct
        self._val_ccs     = np.array(succ_ccs, dtype=float)
        self._adduct_set  = adduct_SET

    # ------------------------------------------------------------------
    # Val RMSE evaluation
    # ------------------------------------------------------------------

    def _compute_val_rmse(self) -> float:
        from sigma.model import predict as sigma_predict
        preds = sigma_predict(self.model, self._adduct_set, self._val_dataset, self._val_adduct)
        preds = np.array(preds, dtype=float)
        return float(np.sqrt(np.mean((preds - self._val_ccs) ** 2)))

    # ------------------------------------------------------------------
    # Per-epoch logic
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        ep = self._epoch_count

        if ep % self.check_val_every_n_epoch != 0:
            return

        if self._val_dataset is None:
            self._build_val_dataset()

        val_rmse = self._compute_val_rmse()
        train_loss = logs.get("loss", float("nan"))

        improved = val_rmse < self._best_val_rmse - self.min_delta
        if improved:
            self.model.save(self._best_model_path)
            self._best_val_rmse = val_rmse
            self._best_epoch = ep
            self.wait = 0
            marker = "  <-- best"
        else:
            self.wait += 1
            marker = ""

        print(f"  [VAL]  epoch {ep:4d}  train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}"
              f"  best={self._best_val_rmse:.4f}  wait={self.wait}/{self.patience}{marker}")

        if self.wait >= self.patience:
            raise EarlyStopSignal(
                f"Val RMSE did not improve for {self.patience} checks "
                f"(best={self._best_val_rmse:.4f} at epoch {self._best_epoch})"
            )
