"""CSV export, ensemble predictions, and cross-fold summaries."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pancdr_pipeline.evaluation import EVAL_NAMES, _metrics_from_predictions


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(str(path), index=False)


def write_json(data, path):
    ensure_dir(Path(path).parent)
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def fold_dir(output_dir, fold_id):
    return Path(output_dir) / "fold_{}".format(fold_id)


def build_ensemble_predictions(output_dir, eval_names, threshold):
    # type: (Path, List[str], float) -> Dict[str, pd.DataFrame]
    output_dir = Path(output_dir)
    summary_dir = ensure_dir(output_dir / "summary")
    ensemble_preds = {}

    for eval_name in eval_names:
        frames = []
        for fold_path in sorted(output_dir.glob("fold_*")):
            pred_path = fold_path / "{}_prediction_results.csv".format(eval_name)
            if not pred_path.is_file():
                continue
            df = pd.read_csv(str(pred_path))
            fold_id = int(fold_path.name.split("_")[1])
            df = df.rename(columns={"score": "fold_{}_score".format(fold_id)})
            keep = ["sample_id", "drug_name", "label", "fold_{}_score".format(fold_id)]
            frames.append(df[keep])

        if not frames:
            continue
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on=["sample_id", "drug_name", "label"], how="outer")

        score_cols = [c for c in merged.columns if c.startswith("fold_") and c.endswith("_score")]
        merged = merged.copy()
        merged["mean_score"] = merged[score_cols].mean(axis=1)
        merged["std_score"] = merged[score_cols].std(axis=1)
        merged["pred_label"] = (merged["mean_score"] >= threshold).astype(int)
        merged.insert(0, "eval_name", eval_name)
        out_path = summary_dir / "{}_ensemble_prediction_results.csv".format(eval_name)
        write_csv(merged, out_path)
        ensemble_preds[eval_name] = merged

        summary, per_drug = _metrics_from_predictions(
            merged.rename(columns={"mean_score": "score"}), threshold, eval_name
        )
        write_csv(summary, summary_dir / "{}_ensemble_metrics_summary.csv".format(eval_name))
        write_csv(per_drug, summary_dir / "{}_ensemble_metrics_per_drug.csv".format(eval_name))

    return ensemble_preds


def build_cross_fold_summary(output_dir, eval_names):
    rows = []
    output_dir = Path(output_dir)
    metric_cols = ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]

    for eval_name in eval_names:
        vals = {m: [] for m in metric_cols}
        for fold_path in sorted(output_dir.glob("fold_*")):
            summary_path = fold_path / "{}_metrics_summary.csv".format(eval_name)
            if not summary_path.is_file():
                continue
            s = pd.read_csv(str(summary_path))
            if s.empty:
                continue
            for m in metric_cols:
                if m in s.columns:
                    v = s.iloc[0][m]
                    if pd.notna(v):
                        vals[m].append(float(v))
        for metric, arr in vals.items():
            if not arr:
                continue
            rows.append(
                {
                    "eval_name": eval_name,
                    "metric": metric,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n_folds": len(arr),
                }
            )
    cross = pd.DataFrame(rows)
    write_csv(cross, output_dir / "summary" / "cross_fold_metrics_summary.csv")
    return cross
