"""K-means clustering on encoder_mu latent (shared cluster model across domains)."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pancdr_pipeline.reports import write_csv


def run_kmeans_for_fold(latent_dir, fold_id, n_clusters=5, seed=0):
    latent_dir = Path(latent_dir)
    mu_files = sorted(latent_dir.glob("*_encoder_mu.csv"))
    if not mu_files:
        return pd.DataFrame(), pd.DataFrame()

    frames = []
    for csv_path in mu_files:
        df = pd.read_csv(str(csv_path))
        df["source_csv"] = csv_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    latent_cols = [c for c in combined.columns if c.startswith("latent_") and c[7:].isdigit()]
    if len(combined) < n_clusters or not latent_cols:
        return pd.DataFrame(), pd.DataFrame()

    X = combined[latent_cols].values
    n_fit = min(n_clusters, len(combined))
    km = KMeans(n_clusters=n_fit, random_state=seed, n_init=10)
    clusters = km.fit_predict(X)

    assignments = combined[["sample_id", "domain", "eval_name"]].copy()
    if "cancer_type" in combined.columns:
        assignments["cancer_type"] = combined["cancer_type"]
    assignments["cluster"] = clusters
    assignments["fold"] = fold_id

    sil = float("nan")
    if len(set(clusters)) > 1:
        try:
            sil = float(silhouette_score(X, clusters))
        except Exception:
            sil = float("nan")

    sizes = {int(k): int(v) for k, v in pd.Series(clusters).value_counts().items()}
    summary = pd.DataFrame(
        [
            {
                "fold": fold_id,
                "eval_name": "combined_encoder_mu",
                "n_clusters": len(sizes),
                "silhouette_score": sil,
                "cluster_size_json": json.dumps(sizes),
                "n_samples": len(combined),
            }
        ]
    )

    write_csv(assignments, latent_dir / "kmeans_assignments.csv")
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
