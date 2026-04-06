"""
Generate 3D conformers for every molecule in data.csv once.

Output: a single .pkl with two keys:
  "conformers" : dict { smiles_str -> list of [x,y,z] per atom (heavy+H),
                        or None if generation failed }
  "All_Atoms"  : sorted list of all element symbols in the dataset

Run once before any experiment:
    source ~/.venvs/SigmaCCS/bin/activate
    python scripts/prepare_conformers.py \
        --data ../GraphCCS/data/data.csv \
        --out  data/conformers.pkl
"""

import argparse
import os
import pickle
import sys

import pandas as pd
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import AllChem

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from sigma.GraphData import GetSmilesAtomSet


def _generate_conformer(smi: str):
    """
    Replicate the per-molecule logic from sigma.GraphData.Generating_coordinates.
    Returns list of [x, y, z] for every atom in the AddHs molecule,
    or None if generation fails for any reason.
    """
    try:
        iMol = Chem.MolFromSmiles(smi)
        if iMol is None:
            return None
        iMol = Chem.RemoveHs(iMol)
        atoms = [atom.GetSymbol() for atom in iMol.GetAtoms()]
        bonds = list(iMol.GetBonds())
        if len(atoms) == 1 and len(bonds) <= 1:
            return None
        iMol3D = Chem.AddHs(iMol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = -1
        ps.maxAttempts = 1
        ps.numThreads = 0
        ps.useRandomCoords = True
        re = AllChem.EmbedMultipleConfs(iMol3D, numConfs=1, params=ps)
        if len(re) == 0:
            return None
        AllChem.MMFFOptimizeMoleculeConfs(iMol3D, numThreads=0)
        coords = [list(iMol3D.GetConformer().GetAtomPosition(atom.GetIdx()))
                  for atom in iMol3D.GetAtoms()]
        return coords
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate 3D conformers for all molecules in data.csv"
    )
    parser.add_argument("--data", required=True,
                        help="Path to data.csv (GraphCCS format: smiles, adducts, label columns)")
    parser.add_argument("--out", required=True,
                        help="Output .pkl path, e.g. data/conformers.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    all_smiles = df[smiles_col].tolist()
    print(f"Loaded {len(all_smiles)} rows from {args.data}")

    # Atom set from the entire dataset — this is the superset used for caching.
    # Individual experiments may use a subset (train-set atoms); the patch in
    # run_experiment.py re-applies the element filter at call time.
    print("Computing atom set...")
    All_Atoms = GetSmilesAtomSet(all_smiles)
    print(f"  {len(All_Atoms)} elements: {All_Atoms}")

    # Deduplicate: same SMILES string → same molecule → same conformer
    unique_smiles = list(dict.fromkeys(all_smiles))
    print(f"Unique SMILES: {len(unique_smiles)}  (total rows: {len(all_smiles)})")

    cache = {}
    failed = []
    for smi in tqdm(unique_smiles, desc="Generating conformers", ncols=90):
        coords = _generate_conformer(smi)
        cache[smi] = coords
        if coords is None:
            failed.append(smi)

    n_ok = len(unique_smiles) - len(failed)
    print(f"\nSucceeded: {n_ok}/{len(unique_smiles)}  |  Failed: {len(failed)}")
    if failed:
        print("Failed SMILES:")
        for s in failed:
            print(f"  {s}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"conformers": cache, "All_Atoms": All_Atoms}
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
