# Drug-PANCDR

Dockerized reproduction of [DMCB-GIST/PANCDR](https://github.com/DMCB-GIST/PANCDR) — precision medicine prediction using adversarial networks for cancer drug response.

## Environment

- Python 3.6
- PyTorch 1.10.1 + CUDA 11.1
- RDKit (`rdkit-pypi==2021.9.4`) for SMILES → drug graph featurization
- See `requirements.txt` and `Dockerfile.pancdr`

## Quick start (Docker)

```bash
# Build image (from this repo root)
docker build -f Dockerfile.pancdr -t pancdr:cuda111 .

# Run container (mount parent Drug dir)
docker run --gpus all -itd \
  --name PANCDR \
  -v "/path/to/Drug":/workspace/ \
  -w /workspace/PANCDR/src \
  pancdr:cuda111
```

Place PANCDR legacy data under `data/GDSC/` and `data/TCGA/` for the original baseline scripts. The new pipeline uses DAPL-master CSVs (see below).

## Multi-TCGA pipeline (`outputs_pancdr_5fold_multitcga`)

Drug-AACDR-style modular pipeline for PANCDR: **5-fold source (CCLE/GDSC) training** with **5 independent TCGA target evaluation sets**, latent export, FID, k-means, and t-SNE.

### Design

| Item | Setting |
|------|---------|
| Source domain | CCLE omics + GDSC2 dose-response (`DAPL-master/data/`) |
| Target domain | TCGA omics (unlabeled in training) + 5 labeled eval CSVs |
| Cross-validation | 5-fold on source pairs; `source_test_size=0.10` held out per fold |
| Model selection | **Source validation AUROC** (early stopping) |
| TCGA labels in training | **No** — labels used only for evaluation |
| Drug graphs | RDKit SMILES → PANCDR graph `[100, 75]` via `drug_graph_adapter.py` |
| Threshold | Fixed **0.5** |
| Hyperparameters | `src/tuned_hyperparameters/TCGA_CV_params.csv` |

Thin CLI entry points:

- `src/pancdr_multilabel_hyper_main.py` — train + evaluate + reports
- `src/pancdr_infer_target_main.py` — inference-only from saved fold checkpoints

Pipeline modules live under `src/pancdr_pipeline/` (`run.py`, `infer.py`, `trainer_wrapper.py`, `evaluation.py`, `latent.py`, `fid.py`, `kmeans.py`, `tsne.py`, …).

### Workspace layout

Mount the parent `Drug/` directory to `/workspace/` in Docker. The pipeline resolves paths relative to `--workspace_root /workspace`:

```
/workspace/
├── DAPL-master/data/
│   ├── pretrain_ccle.csv
│   ├── GDSC2_fitted_dose_response_MaxScreen_raw.csv
│   ├── GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv
│   └── TCGA/
│       ├── pretrain_tcga.csv
│       └── <target eval CSVs — see configs/target_eval_sets.json>
└── PANCDR/
    ├── configs/target_eval_sets.json
    ├── src/
    └── outputs_pancdr_5fold_multitcga/   # default full-run output
```

### Run

```bash
cd /workspace/PANCDR/src

# Full 5-fold training + evaluation + latent/FID/k-means/t-SNE
python pancdr_multilabel_hyper_main.py \
  --workspace_root /workspace \
  --target_eval_config PANCDR/configs/target_eval_sets.json \
  --output_dir PANCDR/outputs_pancdr_5fold_multitcga \
  --n_splits 5 --threshold 0.5 --device cuda

# Background (recommended for full run)
nohup python -u pancdr_multilabel_hyper_main.py \
  --workspace_root /workspace \
  --target_eval_config PANCDR/configs/target_eval_sets.json \
  --output_dir PANCDR/outputs_pancdr_5fold_multitcga \
  --n_splits 5 --threshold 0.5 --device cuda \
  > /workspace/PANCDR/outputs_pancdr_5fold_multitcga/run.log 2>&1 &

# Inference-only — reuses checkpoint config.json; override with flags
python pancdr_infer_target_main.py \
  --checkpoint_dir /workspace/PANCDR/outputs_pancdr_5fold_multitcga \
  --output_dir /workspace/PANCDR/outputs_pancdr_infer \
  --target_eval_config /workspace/PANCDR/configs/target_eval_sets.json \
  --run_latent --run_fid --run_kmeans --run_tsne

# Disable latent steps: --no_latent --no_fid --no_kmeans --no_tsne
```

### Target eval sets (`configs/target_eval_sets.json`)

| Name | DAPL path (under `/workspace/`) |
|------|----------------------------------|
| `target_primary` | `DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `target_only` | `DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `dapl` | `DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv` |
| `target_aacdr` | `DAPL-master/data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` |
| `target_aacdr_only` | `DAPL-master/data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` |

Per-fold evaluation also includes `source_test` (held-out source pairs).

### Key outputs

```
outputs_pancdr_5fold_multitcga/
├── config.json
├── manifest.json
├── hyperparams.json
├── reports/
│   ├── feature_alignment_report.csv
│   ├── drug_index.csv
│   ├── drug_graph_availability_report.csv
│   ├── fold_assignments.csv
│   └── ...
├── fold_0..4/
│   ├── best_model.pt
│   ├── source_test_*.{csv}
│   ├── target_primary_*.{csv}   # + target_only, dapl, target_aacdr, target_aacdr_only
│   └── latent/
│       ├── source_{train,valid,test}_source_encoder_{mu,z}.csv
│       ├── target_unlabeled_encoder_{mu,z}.csv
│       ├── <eval>_target_encoder_{mu,z}.csv
│       ├── fid_summary.csv
│       ├── kmeans_assignments.csv
│       ├── kmeans_summary.csv
│       └── tsne_coordinates.csv
└── summary/
    ├── cross_fold_metrics_summary.csv
    ├── *_ensemble_prediction_results.csv
    ├── *_ensemble_metrics_summary.csv
    ├── fid_cross_fold_summary.csv
    ├── fid_cross_fold_aggregated_summary.csv
    └── kmeans_cross_fold_summary.csv
```

Latent analysis uses **encoder_mu** for FID / k-means / t-SNE; **encoder_z** is exported alongside for all splits.

### Completed run (2026-07-07)

Full 5-fold run completed in container `PANCDR` (`outputs_pancdr_5fold_multitcga/`, ~1.8 GB).

**Source validation AUROC (model selection, per fold):**

| Fold | best_source_valid_auc |
|------|----------------------|
| 0 | 0.894 |
| 1 | 0.879 |
| 2 | 0.895 |
| 3 | 0.888 |
| 4 | 0.888 |

**Cross-fold AUROC (mean ± std, threshold 0.5):**

| Eval set | AUROC | AUPRC |
|----------|-------|-------|
| source_test | 0.891 ± 0.004 | 0.736 ± 0.008 |
| target_primary | 0.567 ± 0.010 | 0.618 ± 0.009 |
| target_only | 0.512 ± 0.036 | 0.656 ± 0.027 |
| dapl | 0.502 ± 0.016 | 0.492 ± 0.013 |
| target_aacdr | 0.610 ± 0.017 | 0.542 ± 0.012 |
| target_aacdr_only | 0.474 ± 0.020 | 0.565 ± 0.027 |

**FID (encoder_mu, source_train → target; 5 folds, 0 NaN):**

| Target | fid_mean ± fid_std |
|--------|-------------------|
| target_unlabeled | 225.1 ± 106.5 |
| dapl | 236.2 ± 123.3 |
| target_aacdr | 268.2 ± 116.8 |
| target_aacdr_only | 261.0 ± 116.4 |
| target_only | 278.1 ± 117.7 |
| target_primary | 289.2 ± 114.6 |

See `summary/cross_fold_metrics_summary.csv`, `summary/fid_cross_fold_aggregated_summary.csv`, and `manifest.json` for full numbers.

### Smoke test

Quick sanity check (1 fold, `--debug_rows 200`):

```bash
python pancdr_multilabel_hyper_main.py \
  --workspace_root /workspace \
  --target_eval_config PANCDR/configs/target_eval_sets.json \
  --output_dir PANCDR/outputs_debug_smoke \
  --debug_rows 200 --device cuda
```

Verify `fold_0/latent/fid_summary.csv` has finite FID values (not all NaN).

## Legacy reproduction workflow

Original baseline scripts are preserved (`run_PANCDR.py`, etc.):

```bash
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh check
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh smoke
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh tcga
```

## Reproduced results (TCGA classification, legacy 100-train)

| Metric | This repo | Paper (Table 1) |
|--------|-----------|------------------|
| Mean AUC | 0.7081 | 0.7106 |
| Std AUC | 0.0265 | 0.0246 |

## Changes from upstream

- Drug-AACDR-style `pancdr_pipeline/` with 5-fold source training and 5 TCGA eval sets
- RDKit-based drug graphs from extended SMILES (no legacy `drug_graph_feat/`)
- Source validation AUROC for model selection; TCGA labels never used in training
- Removed hard-coded `CUDA_VISIBLE_DEVICES` for Docker / single-GPU runs

## Citation

Please cite the original PANCDR paper and repository when using this code.
