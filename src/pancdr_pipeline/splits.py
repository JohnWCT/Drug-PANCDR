"""Grouped source splits by sample_id (no response-row leakage)."""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


@dataclass
class FoldSplit:
    fold: int
    train_sample_ids: List[str]
    valid_sample_ids: List[str]
    source_test_sample_ids: List[str]


@dataclass
class SourceSplits:
    folds: List[FoldSplit]
    source_split: pd.DataFrame
    fold_assignments: pd.DataFrame


def _pseudo_binary_labels(response, sample_ids):
    sid_to_labels = response.groupby("sample_id")["label"].mean()
    pseudo = np.zeros(len(sample_ids), dtype=int)
    for i, sid in enumerate(sample_ids):
        if sid in sid_to_labels.index:
            pseudo[i] = int(sid_to_labels.loc[sid] >= 0.5)
    return pseudo


def build_source_splits(source_response, n_splits, source_test_size, seed):
    # type: (...) -> SourceSplits
    unique_samples = sorted(source_response["sample_id"].astype(str).unique().tolist())
    n = len(unique_samples)
    if n < n_splits + 1:
        raise ValueError("Need more unique samples than folds: n={}, folds={}".format(n, n_splits))

    indices = np.arange(n)
    pseudo = _pseudo_binary_labels(source_response, unique_samples)
    min_test = max(1, int(round(n * source_test_size)))
    eff_test_size = min(min_test / n, 0.5)

    try:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed, stratify=pseudo
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed
        )

    test_ids = [unique_samples[int(i)] for i in sorted(test_idx)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    assignment_rows = []
    split_rows = []

    for fold_id, (tr, va) in enumerate(kf.split(train_val_idx)):
        train_ids = [unique_samples[int(train_val_idx[i])] for i in tr]
        val_ids = [unique_samples[int(train_val_idx[i])] for i in va]
        folds.append(
            FoldSplit(
                fold=fold_id,
                train_sample_ids=train_ids,
                valid_sample_ids=val_ids,
                source_test_sample_ids=test_ids,
            )
        )
        for sid in train_ids:
            assignment_rows.append({"sample_id": sid, "split": "train", "fold": fold_id})
            split_rows.append({"sample_id": sid, "split": "trainval", "fold": fold_id})
        for sid in val_ids:
            assignment_rows.append({"sample_id": sid, "split": "valid", "fold": fold_id})
            split_rows.append({"sample_id": sid, "split": "trainval", "fold": fold_id})
        for sid in test_ids:
            assignment_rows.append({"sample_id": sid, "split": "source_test", "fold": fold_id})
            split_rows.append({"sample_id": sid, "split": "source_test", "fold": -1})

    return SourceSplits(
        folds=folds,
        source_split=pd.DataFrame(split_rows).drop_duplicates(),
        fold_assignments=pd.DataFrame(assignment_rows),
    )
