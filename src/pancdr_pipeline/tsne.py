"""t-SNE coordinates for latent visualization."""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.manifold import TSNE

from pancdr_pipeline.reports import write_csv


def run_tsne_for_fold(latent_dir, fold_id, seed=0, max_samples=2000):
    latent_dir = Path(latent_dir)
    mu_files = list(latent_dir.glob("*_encoder_mu.csv"))
    if not mu_files:
        return pd.DataFrame()

    frames = []
    for path in mu_files:
        frames.append(pd.read_csv(str(path)))
    combined = pd.concat(frames, ignore_index=True)
    latent_cols = [c for c in combined.columns if c.startswith("latent_")]
    if len(combined) < 5 or not latent_cols:
        warnings.warn("Fold {}: insufficient samples for t-SNE".format(fold_id))
        return pd.DataFrame()

    if len(combined) > max_samples:
        combined = combined.sample(n=max_samples, random_state=seed)

    X = combined[latent_cols].values
    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X) - 1))
        coords = tsne.fit_transform(X)
    except Exception as exc:
        warnings.warn("Fold {} t-SNE skipped: {}".format(fold_id, exc))
        return pd.DataFrame()

    out = combined[["sample_id", "domain", "eval_name"]].copy()
    if "cancer_type" in combined.columns:
        out["cancer_type"] = combined["cancer_type"]
    out["tsne_1"] = coords[:, 0]
    out["tsne_2"] = coords[:, 1]
    write_csv(out, latent_dir / "tsne_coordinates.csv")
    return out
