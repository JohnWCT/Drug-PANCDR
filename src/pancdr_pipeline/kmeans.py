"""K-means clustering on encoder_mu latent."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pancdr_pipeline.reports import write_csv


def run_kmeans_for_fold(latent_dir, fold_id, n_clusters=5, seed=0):
    latent_dir = Path(latent_dir)
    assignment_frames = []
    summary_rows = []

    for csv_path in sorted(latent_dir.glob("*_encoder_mu.csv")):
        df = pd.read_csv(str(csv_path))
        latent_cols = [c for c in df.columns if c.startswith("latent_")]
        if len(df) < n_clusters or not latent_cols:
            continue
        X = df[latent_cols].values
        km = KMeans(n_clusters=min(n_clusters, len(df)), random_state=seed, n_init=10)
        clusters = km.fit_predict(X)
        out = df[["sample_id", "domain", "eval_name"]].copy()
        if "cancer_type" in df.columns:
            out["cancer_type"] = df["cancer_type"]
        out["cluster"] = clusters
        assignment_frames.append(out)

        sil = float("nan")
        if len(set(clusters)) > 1:
            try:
                sil = float(silhouette_score(X, clusters))
            except Exception:
                sil = float("nan")
        sizes = {int(k): int(v) for k, v in pd.Series(clusters).value_counts().items()}
        summary_rows.append(
            {
                "fold": fold_id,
                "eval_name": df["eval_name"].iloc[0],
                "n_clusters": len(sizes),
                "silhouette_score": sil,
                "cluster_size_json": json.dumps(sizes),
            }
        )

    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not assignments.empty:
        write_csv(assignments, latent_dir / "kmeans_assignments.csv")
    if not summary.empty:
        write_csv(summary, latent_dir / "kmeans_summary.csv")
    return assignments, summary


def build_cross_fold_kmeans_summary(output_dir):
    frames = []
    output_dir = Path(output_dir)
    for fold_path in sorted(output_dir.glob("fold_*")):
        summary_path = fold_path / "latent" / "kmeans_summary.csv"
        if summary_path.is_file():
            frames.append(pd.read_csv(str(summary_path)))
    if frames:
        cross = pd.concat(frames, ignore_index=True)
        write_csv(cross, output_dir / "summary" / "kmeans_cross_fold_summary.csv")
        return cross
    return pd.DataFrame()
