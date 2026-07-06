"""
PANCDR smoke test: 與 run_PANCDR.py 相同流程，但只訓練 1 次。

目的:
  - 驗證 GDSC/TCGA 資料可正確載入與特徵化
  - 驗證 discriminator + CDR 模型可在 GPU 上完成至少一輪 early-stopping 訓練
  - 避免直接跑 100 次（數小時）才發現環境或路徑問題

輸出:
  - TCGA_smoke_results.csv  (單次 AUC + mean 列)
  - ../checkpoint/TCGA/0_model.pt
"""
import torch
import random
import os
import numpy as np
import pandas as pd

from utils import DataGenerate, DataFeature
from ModelTraining.PANCDR import train_PANCDR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DPATH = '../data'
Drug_info_file = '%s/GDSC/GDSC_drug_binary.csv' % DPATH
Cell_line_info_file = '%s/GDSC/Cell_Lines_Details.txt' % DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Cancer_response_exp_file = '%s/GDSC/GDSC_binary_response_151.csv' % DPATH
Gene_expression_file = '%s/GDSC/GDSC_expr_z_702.csv' % DPATH
P_Gene_expression_file = '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv' % DPATH

T_Drug_info_file = '%s/TCGA/TCGA_drug_new.csv' % DPATH
T_Patient_info_file = '%s/TCGA/TCGA_type_new.txt' % DPATH
T_Drug_feature_file = '%s/TCGA/drug_graph_feat' % DPATH
T_Cancer_response_exp_file = '%s/TCGA/TCGA_response_new.csv' % DPATH
T_Gene_expression_file = '%s/TCGA/TCGA_expr_z_702.csv' % DPATH

SMOKE_ITERS = int(os.environ.get('PANCDR_SMOKE_ITERS', '1'))

if __name__ == '__main__':
    print('device:', device)
    print('smoke iterations:', SMOKE_ITERS)

    drug_feature, gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(
        Drug_info_file, Cell_line_info_file, Drug_feature_file,
        Gene_expression_file, P_Gene_expression_file, Cancer_response_exp_file,
    )
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(
        T_Drug_info_file, T_Patient_info_file, T_Drug_feature_file,
        T_Gene_expression_file, None, T_Cancer_response_exp_file, dataset="TCGA",
    )
    TX_drug_data_test, TX_gexpr_data_test, TY_test, _ = DataFeature(
        T_data_idx, T_drug_feature, T_gexpr_feature, dataset="TCGA",
    )

    TX_drug_feat_data_test = np.array([item[0] for item in TX_drug_data_test])
    TX_drug_adj_data_test = np.array([item[1] for item in TX_drug_data_test])
    TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
    TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
    TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
    TY_test = torch.FloatTensor(TY_test).to(device)

    X_drug_data, X_gexpr_data, Y, _ = DataFeature(data_idx, drug_feature, gexpr_feature)
    X_drug_feat_data = np.array([item[0] for item in X_drug_data])
    X_drug_adj_data = np.array([item[1] for item in X_drug_data])

    train_data = [X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature]
    test_data = [TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test]

    df = pd.read_csv("tuned_hyperparameters/TCGA_CV_params.csv")
    best_params = eval(
        df.loc[(df["Model"] == "PANCDR") & (df["Classification"] == "T"), "Best_params"].values[0]
    )
    model = train_PANCDR(train_data, test_data)

    results = []
    print("Smoke training ...")
    for iter_idx in range(SMOKE_ITERS):
        weight_path = '../checkpoint/TCGA/%d_model.pt' % iter_idx
        auc_tcga = model.train(best_params, weight_path)
        print('iter %d - roc-TCGA: %.4f' % (iter_idx, auc_tcga))
        results.append(auc_tcga)

    result_df = pd.DataFrame(results, columns=['TCGA AUC'])
    result_df.loc['mean',] = result_df.mean().values
    result_df.to_csv('TCGA_smoke_results.csv')
    print('wrote TCGA_smoke_results.csv')
