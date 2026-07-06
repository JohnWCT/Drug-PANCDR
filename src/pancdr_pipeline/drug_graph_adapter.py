"""Build PANCDR-compatible drug graphs from SMILES using RDKit."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pancdr_pipeline.drug_index import DrugIndexResult

try:
    from rdkit import Chem
except ImportError:
    Chem = None

MAX_ATOMS = 100
ATOM_FEAT_DIM = 75


@dataclass
class DrugGraphBundle:
    graphs: Dict[str, Tuple[np.ndarray, np.ndarray]]
    availability_report: pd.DataFrame
    edge_report: pd.DataFrame
    atom_feature_dim: int


def _require_rdkit():
    if Chem is None:
        raise ImportError(
            "RDKit is required for PANCDR drug graph generation. "
            "Install rdkit-pypi in the PANCDR Docker image."
        )


def _one_hot(value, choices):
    arr = np.zeros(len(choices), dtype=np.float32)
    try:
        arr[choices.index(value)] = 1.0
    except ValueError:
        pass
    return arr


def _atom_features(atom):
    # 75-dim ConvMol-compatible atom feature vector (RDKit only)
    from rdkit import Chem as RDKitChem

    atomic_num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization = int(atom.GetHybridization())
    is_aromatic = int(atom.GetIsAromatic())
    total_h = atom.GetTotalNumHs()
    chiral = int(atom.GetChiralTag())

  # Pad/truncate to 75 dims with common descriptors
    feats = []
    feats.extend(_one_hot(atomic_num, list(range(1, 61))))
    feats.extend(_one_hot(degree, list(range(0, 7))))
    feats.extend(_one_hot(formal_charge, list(range(-2, 4))))
    feats.extend(_one_hot(hybridization, list(range(1, 8))))
    feats.append(float(is_aromatic))
    feats.extend(_one_hot(total_h, list(range(0, 5))))
    feats.extend(_one_hot(chiral, list(range(0, 4))))
    feats = np.array(feats, dtype=np.float32)
    if feats.shape[0] < ATOM_FEAT_DIM:
        feats = np.pad(feats, (0, ATOM_FEAT_DIM - feats.shape[0]))
    else:
        feats = feats[:ATOM_FEAT_DIM]
    return feats


def _smiles_to_graph(smiles):
    # type: (str) -> Tuple[np.ndarray, np.ndarray, int, int]
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: {}".format(smiles))

    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_ATOMS:
        raise ValueError("Molecule has {} atoms > MAX_ATOMS={}".format(n_atoms, MAX_ATOMS))

    feat_mat = np.zeros((n_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        feat_mat[i] = _atom_features(atom)

    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    n_edges = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        n_edges += 2
    np.fill_diagonal(adj, 1.0)

    from utils import CalculateGraphFeat

    adj_list = []
    for i in range(n_atoms):
        neighbors = [int(x) for x in np.where(adj[i] > 0)[0] if x != i]
        adj_list.append(neighbors)
    drug_feat, drug_adj = CalculateGraphFeat(feat_mat, adj_list)
    return drug_feat.astype(np.float32), drug_adj.astype(np.float32), n_atoms, n_edges


def build_drug_graphs(drug_index_result, output_dir, force_rebuild=False):
    # type: (DrugIndexResult, Path, bool) -> DrugGraphBundle
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "drug_graph_cache.pkl"

    if cache_path.is_file() and not force_rebuild:
        with open(str(cache_path), "rb") as f:
            cached = pickle.load(f)
        return DrugGraphBundle(**cached)

    graphs = {}
    avail_rows = []
    edge_rows = []
    smiles_map = drug_index_result.smiles_map

    for drug_key in drug_index_result.drug_keys:
        has_smiles = drug_key in smiles_map
        ok = False
        err = ""
        n_atoms = 0
        n_edges = 0
        try:
            if not has_smiles:
                raise KeyError("missing SMILES")
            drug_feat, drug_adj, n_atoms, n_edges = _smiles_to_graph(smiles_map[drug_key])
            graphs[drug_key] = (drug_feat, drug_adj)
            ok = True
        except Exception as exc:
            err = str(exc)

        avail_rows.append(
            {
                "drug_key": drug_key,
                "has_smiles": int(has_smiles),
                "has_graph": int(ok),
                "error": err,
            }
        )
        edge_rows.append(
            {
                "drug_key": drug_key,
                "n_atoms": int(n_atoms),
                "n_edges": int(n_edges),
                "atom_feature_dim": ATOM_FEAT_DIM if ok else 0,
            }
        )

    availability_report = pd.DataFrame(avail_rows)
    edge_report = pd.DataFrame(edge_rows)
    bundle = DrugGraphBundle(
        graphs=graphs,
        availability_report=availability_report,
        edge_report=edge_report,
        atom_feature_dim=ATOM_FEAT_DIM,
    )
    with open(str(cache_path), "wb") as f:
        pickle.dump(
            {
                "graphs": bundle.graphs,
                "availability_report": bundle.availability_report,
                "edge_report": bundle.edge_report,
                "atom_feature_dim": bundle.atom_feature_dim,
            },
            f,
        )
    return bundle
