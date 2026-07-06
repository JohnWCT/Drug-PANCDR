"""Build fold-level tensor bundles for PANCDR trainer."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pancdr_pipeline.drug_graph_adapter import DrugGraphBundle
from pancdr_pipeline.splits import FoldSplit


@dataclass
class LabeledSplitData:
    drug_feat: np.ndarray
    drug_adj: np.ndarray
    gexpr: np.ndarray
    labels: np.ndarray
    sample_ids: List[str]
    drug_names: List[str]
    drug_keys: List[str]
    cancer_types: List[str]


@dataclass
class TargetUnlabeledData:
    gexpr: np.ndarray
    sample_ids: List[str]


@dataclass
class FoldDataBundle:
    fold_id: int
    source_train: LabeledSplitData
    source_valid: LabeledSplitData
    source_test: LabeledSplitData
    target_unlabeled: TargetUnlabeledData
    target_eval_sets: Dict[str, LabeledSplitData]
    common_features: List[str]


def _omics_row(omics_df, sample_id, feature_names):
    row = omics_df.loc[omics_df["sample_id"].astype(str) == str(sample_id), feature_names]
    if row.empty:
        return None
    return row.iloc[0].astype(float).values


def _build_labeled_split(
    response_df,
    omics_df,
    feature_names,
    drug_graphs,
    allowed_samples=None,
    max_rows=None,
):
    # type: (...) -> LabeledSplitData
    rows = []
    for _, r in response_df.iterrows():
        sid = str(r["sample_id"])
        if allowed_samples is not None and sid not in allowed_samples:
            continue
        drug_key = str(r["drug_key"])
        if drug_key not in drug_graphs:
            continue
        gexpr = _omics_row(omics_df, sid, feature_names)
        if gexpr is None:
            continue
        drug_feat, drug_adj = drug_graphs[drug_key]
        rows.append(
            {
                "drug_feat": drug_feat,
                "drug_adj": drug_adj,
                "gexpr": gexpr,
                "label": float(r["label"]),
                "sample_id": sid,
                "drug_name": r.get("drug_name", drug_key),
                "drug_key": drug_key,
                "cancer_type": str(r.get("cancer_type", "")),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        return LabeledSplitData(
            drug_feat=np.zeros((0, 100, 75)),
            drug_adj=np.zeros((0, 100, 100)),
            gexpr=np.zeros((0, len(feature_names))),
            labels=np.zeros((0,)),
            sample_ids=[],
            drug_names=[],
            drug_keys=[],
            cancer_types=[],
        )

    return LabeledSplitData(
        drug_feat=np.stack([x["drug_feat"] for x in rows]),
        drug_adj=np.stack([x["drug_adj"] for x in rows]),
        gexpr=np.stack([x["gexpr"] for x in rows]),
        labels=np.array([x["label"] for x in rows], dtype=np.float32),
        sample_ids=[x["sample_id"] for x in rows],
        drug_names=[x["drug_name"] for x in rows],
        drug_keys=[x["drug_key"] for x in rows],
        cancer_types=[x["cancer_type"] for x in rows],
    )


def build_fold_data_bundle(
    fold_split,
    source_response,
    source_omics,
    target_omics,
    target_eval_sets,
    drug_graph_bundle,
    common_features,
    debug_rows=None,
):
    # type: (...) -> FoldDataBundle
    graphs = drug_graph_bundle.graphs
    train_set = set(fold_split.train_sample_ids)
    val_set = set(fold_split.valid_sample_ids)
    test_set = set(fold_split.source_test_sample_ids)

    per_fold_cap = None
    if debug_rows is not None:
        per_fold_cap = max(10, debug_rows // 5)

    source_train = _build_labeled_split(
        source_response, source_omics, common_features, graphs, train_set, per_fold_cap
    )
    source_valid = _build_labeled_split(
        source_response, source_omics, common_features, graphs, val_set, per_fold_cap
    )
    source_test = _build_labeled_split(
        source_response, source_omics, common_features, graphs, test_set, per_fold_cap
    )

    unlabeled_rows = []
    for sid in target_omics["sample_id"].astype(str).tolist():
        gexpr = _omics_row(target_omics, sid, common_features)
        if gexpr is not None:
            unlabeled_rows.append((sid, gexpr))
    if debug_rows is not None and len(unlabeled_rows) > debug_rows:
        unlabeled_rows = unlabeled_rows[:debug_rows]

    target_unlabeled = TargetUnlabeledData(
        gexpr=np.stack([x[1] for x in unlabeled_rows]) if unlabeled_rows else np.zeros((0, len(common_features))),
        sample_ids=[x[0] for x in unlabeled_rows],
    )

    eval_bundles = {}
    for name, resp in target_eval_sets.items():
        cap = debug_rows if debug_rows is not None else None
        eval_bundles[name] = _build_labeled_split(
            resp, target_omics, common_features, graphs, allowed_samples=None, max_rows=cap
        )

    return FoldDataBundle(
        fold_id=fold_split.fold,
        source_train=source_train,
        source_valid=source_valid,
        source_test=source_test,
        target_unlabeled=target_unlabeled,
        target_eval_sets=eval_bundles,
        common_features=common_features,
    )
