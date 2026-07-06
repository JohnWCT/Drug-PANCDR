# Drug-PANCDR

Dockerized reproduction of [DMCB-GIST/PANCDR](https://github.com/DMCB-GIST/PANCDR) — precision medicine prediction using adversarial networks for cancer drug response.

## Environment

- Python 3.6
- PyTorch 1.10.1 + CUDA 11.1
- See `requirements.txt` and `Dockerfile.pancdr`

## Quick start (Docker)

```bash
# Build image (from this repo root)
docker build -f Dockerfile.pancdr -t pancdr:cuda111 .

# Run container (mount parent Drug dir or this repo)
docker run --gpus all -itd \
  --name PANCDR \
  -v "/path/to/Drug":/workspace/ \
  -w /workspace/PANCDR/src \
  pancdr:cuda111
```

Place PANCDR data under `data/GDSC/` and `data/TCGA/` as described in the [upstream README](https://github.com/DMCB-GIST/PANCDR). If only `Pretrain_TCGA_expr_702_01A.zip` is available, unzip it to `data/TCGA/Pretrain_TCGA_expr_702_01A.csv`.

## Reproduction workflow

```bash
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh check
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh smoke
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh tcga      # 100-train TCGA
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh baseline  # DeepCDR
docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh nested    # GDSC nested CV
```

## Reproduced results (TCGA classification)

| Metric | This repo | Paper (Table 1) |
|--------|-----------|------------------|
| Mean AUC | 0.7081 | 0.7106 |
| Std AUC | 0.0265 | 0.0246 |

Output: `src/TCGA_100train_results.csv`

## Changes from upstream

- Removed hard-coded `CUDA_VISIBLE_DEVICES` for Docker / single-GPU runs
- Added `scripts/reproduce_pancdr.sh` and `src/run_PANCDR_smoke.py`
- Device selection: `cuda` if available, else `cpu`

## Citation

Please cite the original PANCDR paper and repository when using this code.
