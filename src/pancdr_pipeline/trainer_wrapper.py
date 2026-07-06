"""5-fold PANCDR trainer – source validation AUROC only, no TCGA labels in training."""

import itertools
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from pancdr_pipeline.config import PANCDRPipelineConfig
from pancdr_pipeline.datasets import FoldDataBundle
from pancdr_pipeline.model_adapter import PANCDRModelAdapter


@dataclass
class FoldTrainingResult:
    fold_id: int
    checkpoint_path: str
    best_source_valid_auc: float
    best_epoch: int


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_nan(pred):
    return torch.isnan(pred).sum() > 0


def train_one_fold(fold_id, bundle, params, config, output_dir):
    # type: (int, FoldDataBundle, Dict[str, Any], PANCDRPipelineConfig, Path) -> FoldTrainingResult
    _set_seed(config.seed + fold_id)
    device = config.device

    nz = params["nz"]
    d_dim = params["d_dim"]
    lr = params["lr"]
    lr_adv = params["lr_adv"]
    lam = params["lam"]
    batch_size = params["batch_size"]

    tr = bundle.source_train
    va = bundle.source_valid
    tu = bundle.target_unlabeled

    if len(tr.labels) == 0:
        raise ValueError("Fold {} source_train is empty".format(fold_id))

    atom_dim = tr.drug_feat.shape[2]
    adapter = PANCDRModelAdapter(tr.gexpr.shape[1], atom_dim, params, device).build_models()

    X_drug_feat_train = torch.FloatTensor(tr.drug_feat)
    X_drug_adj_train = torch.FloatTensor(tr.drug_adj)
    X_gexpr_train = torch.FloatTensor(tr.gexpr)
    Y_train = torch.FloatTensor(tr.labels)

    X_drug_feat_val = torch.FloatTensor(va.drug_feat).to(adapter.device)
    X_drug_adj_val = torch.FloatTensor(va.drug_adj).to(adapter.device)
    X_gexpr_val = torch.FloatTensor(va.gexpr).to(adapter.device)
    Y_val = torch.FloatTensor(va.labels).to(adapter.device)

    X_t_gexpr = torch.FloatTensor(tu.gexpr)

    gdsc_loader = DataLoader(
        TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train),
        batch_size=batch_size[0],
        shuffle=True,
        drop_last=len(tr.labels) > batch_size[0],
    )
    t_loader = DataLoader(
        TensorDataset(X_t_gexpr),
        batch_size=min(batch_size[1], max(1, len(tu.sample_ids))),
        shuffle=True,
        drop_last=len(tu.sample_ids) > batch_size[1],
    )

    optimizer = torch.optim.Adam(
        itertools.chain(adapter.encoder.parameters(), adapter.gcn.parameters()), lr=lr
    )
    optimizer_adv = torch.optim.Adam(adapter.adv.parameters(), lr=lr_adv)
    loss_fn = torch.nn.BCELoss()

    wait = 0
    best_auc = 0.0
    best_epoch = -1
    ckpt_path = str(output_dir / "fold_{}".format(fold_id) / "best_model.pt")
    output_dir.joinpath("fold_{}".format(fold_id)).mkdir(parents=True, exist_ok=True)

    for epoch in range(config.max_epochs):
        for data in zip(gdsc_loader, cycle(t_loader)):
            drug_feat, drug_adj, gexpr, y_true = data[0]
            t_gexpr = data[1][0]
            drug_feat = drug_feat.to(adapter.device)
            drug_adj = drug_adj.to(adapter.device)
            gexpr = gexpr.to(adapter.device)
            y_true = y_true.view(-1, 1).to(adapter.device)
            t_gexpr = t_gexpr.to(adapter.device)

            adapter.encoder.train()
            adapter.gcn.train()
            adapter.adv.train()

            optimizer_adv.zero_grad()
            F_gexpr, _, _ = adapter.encoder(gexpr)
            F_t_gexpr, _, _ = adapter.encoder(t_gexpr)
            F_g_t_gexpr = torch.cat((F_gexpr, F_t_gexpr))
            z_true = torch.cat(
                (
                    torch.zeros(F_gexpr.shape[0], device=adapter.device),
                    torch.ones(F_t_gexpr.shape[0], device=adapter.device),
                )
            ).view(-1, 1)
            z_pred = adapter.adv(F_g_t_gexpr)
            if _is_nan(z_pred):
                break
            adv_loss = loss_fn(z_pred, z_true)
            adv_loss.backward()
            optimizer_adv.step()

            optimizer.zero_grad()
            g_latents, _, _ = adapter.encoder(gexpr)
            t_latents, _, _ = adapter.encoder(t_gexpr)
            F_g_t_latents = torch.cat((g_latents, t_latents))
            z_true_ = torch.cat(
                (
                    torch.ones(g_latents.shape[0], device=adapter.device),
                    torch.zeros(t_latents.shape[0], device=adapter.device),
                )
            ).view(-1, 1)
            z_pred_ = adapter.adv(F_g_t_latents)
            y_pred = adapter.gcn(drug_feat, drug_adj, g_latents)
            if _is_nan(z_pred_) or _is_nan(y_pred):
                break
            adv_loss_ = loss_fn(z_pred_, z_true_)
            cdr_loss = loss_fn(y_pred, y_true)
            total = cdr_loss + lam * adv_loss_
            total.backward()
            optimizer.step()

        with torch.no_grad():
            adapter.encoder.eval()
            adapter.gcn.eval()
            F_gexpr_val, _, _ = adapter.encoder(X_gexpr_val)
            y_pred_val = adapter.gcn(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
            if len(np.unique(va.labels)) < 2:
                auc_val = 0.0
            else:
                auc_val = roc_auc_score(
                    Y_val.cpu().numpy(), y_pred_val.cpu().detach().numpy()
                )

        if auc_val >= best_auc:
            wait = 0
            best_auc = auc_val
            best_epoch = epoch
            adapter.save_checkpoint(
                ckpt_path,
                {
                    "fold_id": fold_id,
                    "best_source_valid_auc": float(best_auc),
                    "threshold": config.threshold,
                    "common_features": bundle.common_features,
                },
            )
        else:
            wait += 1
            if wait >= config.early_stop_patience:
                break

    return FoldTrainingResult(
        fold_id=fold_id,
        checkpoint_path=ckpt_path,
        best_source_valid_auc=float(best_auc),
        best_epoch=best_epoch,
    )
