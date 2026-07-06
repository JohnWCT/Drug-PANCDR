"""Prediction tables and metrics for source test + 5 TCGA eval sets."""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pancdr_pipeline.config import PANCDRPipelineConfig
from pancdr_pipeline.datasets import FoldDataBundle, LabeledSplitData
from pancdr_pipeline.drug_index import DrugIndexResult
from pancdr_pipeline.model_adapter import PANCDRModelAdapter

EVAL_NAMES = [
    "source_test",
    "target_primary",
    "target_only",
    "dapl",
    "target_aacdr",
    "target_aacdr_only",
]


def _safe_auc(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _safe_auprc(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, scores))
    except ValueError:
        return float("nan")


def _predict_split(adapter, split_data, threshold):
    # type: (PANCDRModelAdapter, LabeledSplitData, float) -> pd.DataFrame
    if len(split_data.labels) == 0:
        return pd.DataFrame()
    drug_feat = torch.FloatTensor(split_data.drug_feat).to(adapter.device)
    drug_adj = torch.FloatTensor(split_data.drug_adj).to(adapter.device)
    gexpr = torch.FloatTensor(split_data.gexpr).to(adapter.device)
    scores = adapter.predict(drug_feat, drug_adj, gexpr).cpu().numpy()
    rows = []
    for i in range(len(scores)):
        score = float(scores[i])
        rows.append(
            {
                "sample_id": split_data.sample_ids[i],
                "drug_name": split_data.drug_names[i],
                "label": int(split_data.labels[i]),
                "score": score,
                "pred_label": int(score >= threshold),
                "cancer_type": split_data.cancer_types[i] if i < len(split_data.cancer_types) else "",
            }
        )
    return pd.DataFrame(rows)


def _metrics_from_predictions(pred_df, threshold, eval_name):
    if pred_df.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "eval_name": eval_name,
                        "n_rows": 0,
                        "n_positive": 0,
                        "n_negative": 0,
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                        "accuracy": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "f1": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "threshold": threshold,
                    }
                ]
            ),
            pd.DataFrame(),
        )

    y = pred_df["label"].astype(int).values
    scores = pred_df["score"].astype(float).values
    pred = pred_df["pred_label"].astype(int).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = pd.DataFrame(
            [
                {
                    "eval_name": eval_name,
                    "n_rows": len(pred_df),
                    "n_positive": int((y == 1).sum()),
                    "n_negative": int((y == 0).sum()),
                    "auroc": _safe_auc(y, scores),
                    "auprc": _safe_auprc(y, scores),
                    "accuracy": float(accuracy_score(y, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "precision": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "threshold": threshold,
                }
            ]
        )

    per_drug_rows = []
    for drug_name, sub in pred_df.groupby("drug_name"):
        yy = sub["label"].astype(int).values
        ss = sub["score"].astype(float).values
        pp = sub["pred_label"].astype(int).values
        per_drug_rows.append(
            {
                "drug_name": drug_name,
                "n_rows": len(sub),
                "n_positive": int((yy == 1).sum()),
                "n_negative": int((yy == 0).sum()),
                "auroc": _safe_auc(yy, ss),
                "auprc": _safe_auprc(yy, ss),
                "accuracy": float(accuracy_score(yy, pp)) if len(yy) else float("nan"),
                "f1": float(f1_score(yy, pp, zero_division=0)) if len(yy) else float("nan"),
                "precision": float(precision_score(yy, pp, zero_division=0)) if len(yy) else float("nan"),
                "recall": float(recall_score(yy, pp, zero_division=0)) if len(yy) else float("nan"),
            }
        )
    return summary, pd.DataFrame(per_drug_rows)


def evaluate_fold(adapter, bundle, drug_index_result, config, fold_id):
    # type: (...) -> Dict[str, Dict[str, pd.DataFrame]]
    threshold = config.threshold
    results = {}

    eval_map = {"source_test": bundle.source_test}
    eval_map.update(bundle.target_eval_sets)

    for eval_name in EVAL_NAMES:
        if eval_name not in eval_map:
            continue
        split_data = eval_map[eval_name]
        pred = _predict_split(adapter, split_data, threshold)
        if pred.empty:
            continue
        pred.insert(0, "fold", fold_id)
        pred.insert(1, "eval_name", eval_name)
        target_only_drugs = set(
            drug_index_result.drug_index.loc[
                drug_index_result.drug_index["is_target_only_drug"] == 1, "drug_key"
            ].astype(str)
        )
        pred["is_target_only_drug"] = pred["drug_name"].str.lower().isin(target_only_drugs).astype(int)
        pred["has_smiles"] = pred["drug_name"].str.lower().map(
            lambda x: int(x in drug_index_result.smiles_map)
        )
        pred["has_graph"] = pred["drug_name"].str.lower().map(
            lambda x: int(x in drug_index_result.smiles_map)
        )
        summary, per_drug = _metrics_from_predictions(pred, threshold, eval_name)
        results[eval_name] = {
            "predictions": pred,
            "summary": summary,
            "per_drug": per_drug,
        }
    return results
