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

## New pipeline: 5-fold + 5 TCGA eval sets

Thin CLI wrappers (Drug-AACDR style):

```bash
cd /workspace/PANCDR/src

# Training + evaluation + latent/FID reports
python pancdr_multilabel_hyper_main.py \
  --workspace_root /workspace \
  --target_eval_config PANCDR/configs/target_eval_sets.json \
  --output_dir PANCDR/outputs_pancdr_5fold_multitcga \
  --n_splits 5 --threshold 0.5 --device cuda

# Inference-only (optionally with latent/FID)
python pancdr_infer_target_main.py \
  --checkpoint_dir /workspace/PANCDR/outputs_pancdr_5fold_multitcga \
  --output_dir /workspace/PANCDR/outputs_pancdr_infer \
  --target_eval_config /workspace/PANCDR/configs/target_eval_sets.json \
  --run_latent --run_fid
```

### Target eval sets (`configs/target_eval_sets.json`)

- `target_primary`
- `target_only`
- `dapl`
- `target_aacdr`
- `target_aacdr_only`

### Key outputs

```
outputs_*/
├── config.json
├── manifest.json
├── reports/
├── fold_0..4/
│   ├── best_model.pt
│   ├── *_prediction_results.csv
│   ├── *_metrics_summary.csv
│   └── latent/
│       ├── *_encoder_mu.csv / *_encoder_z.csv
│       ├── fid_summary.csv
│       ├── kmeans_assignments.csv
│       └── tsne_coordinates.csv
└── summary/
    ├── *_ensemble_prediction_results.csv
    ├── cross_fold_metrics_summary.csv
    ├── fid_cross_fold_summary.csv
    └── fid_cross_fold_aggregated_summary.csv
```

### Smoke test

```bash
python pancdr_multilabel_hyper_main.py \
  --workspace_root /workspace \
  --target_eval_config PANCDR/configs/target_eval_sets.json \
  --output_dir PANCDR/outputs_debug_pancdr \
  --debug_rows 200 --device cuda
```

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
