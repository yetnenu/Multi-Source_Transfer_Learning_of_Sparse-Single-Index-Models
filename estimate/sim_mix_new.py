#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
半监督仿真实验
使用 KEF（核特征估计）进行得分函数估计，强制使用无标签数据。
所有方法（Lasso 等）均使用原始数据，无外部标准化。
KEF 内部自行标准化并还原梯度。
支持断点续传。
评估方法：Lasso（全量数据）、GLMtrans（全量数据，由开关控制）、随机方向训练、固定神经网络。
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
import pickle as pkl
from scipy.special import erf, expit

# ---------- KEF 导入 ----------
try:
    from kscore.estimators import Tikhonov
    from kscore.kernels import CurlFreeIMQ
except ImportError:
    raise ImportError("请安装 numethod 包 (pip install numethod)")

# ---------- GLMtrans 相关 ----------
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from Frank_Wolfe_Algorithm import frank_wolfe_spherical_combination

def install_glmtrans():
    try:
        return importr('glmtrans')
    except:
        print("glmtrans not found. Attempting to install...")
        try:
            utils = importr('utils')
            utils.install_packages('glmtrans')
            return importr('glmtrans')
        except:
            print("Failed to install from CRAN. Trying GitHub...")
            try:
                devtools = importr('devtools', quiet=True)
                devtools.install_github('lianmingli/GLMtrans')
                return importr('glmtrans')
            except Exception as e:
                print(f"Failed to install glmtrans: {e}")
                return None

class GLMTransWrapper:
    def __init__(self):
        self.glmtrans = install_glmtrans()
        if self.glmtrans is not None:
            self.stats = importr('stats')
            self.base = importr('base')
            print("glmtrans package loaded")
        else:
            print("glmtrans not available")

    def fit(self, X_target, y_target, X_source_list=None, y_source_list=None, 
            family="gaussian", method="transfer", transfer_source_id="all", **kwargs):
        if self.glmtrans is None:
            raise RuntimeError("glmtrans not available")
        X_target = np.array(X_target)
        y_target = np.array(y_target)
        with localconverter(numpy2ri.converter):
            r_X_target = ro.r.matrix(X_target, nrow=X_target.shape[0], ncol=X_target.shape[1])
            r_y_target = ro.FloatVector(y_target)
            if X_source_list is not None and y_source_list is not None:
                if len(X_source_list) != len(y_source_list):
                    raise ValueError("X_source_list and y_source_list must have the same length")
                r_source_list = ro.ListVector({
                    f'source{i}': ro.ListVector({
                        'x': ro.r.matrix(np.array(X_source), nrow=X_source.shape[0], ncol=X_source.shape[1]),
                        'y': ro.FloatVector(np.array(y_source))
                    }) for i, (X_source, y_source) in enumerate(zip(X_source_list, y_source_list))
                })
            else:
                r_source_list = ro.NULL
            if transfer_source_id == "all":
                r_transfer_source_id = ro.IntVector(range(1, len(X_source_list)+1)) if X_source_list else ro.NULL
            elif isinstance(transfer_source_id, (list, tuple, np.ndarray)):
                r_transfer_source_id = ro.IntVector(transfer_source_id)
            else:
                r_transfer_source_id = ro.IntVector([transfer_source_id])
            self.model_ = self.glmtrans.glmtrans(
                target=ro.ListVector({'x': r_X_target, 'y': r_y_target}),
                source=r_source_list,
                family=family,
                method=method,
                transfer_source_id=r_transfer_source_id,
                **kwargs
            )
        self.coef_ = np.array(self.model_['beta'])
        return self

    def predict(self, X_new):
        beta = self.coef_
        if X_new.shape[1] == len(beta) - 1:
            X_new_with_intercept = np.column_stack([np.ones(X_new.shape[0]), X_new])
        else:
            X_new_with_intercept = X_new
        return X_new_with_intercept @ beta

    def get_coefficients(self):
        if hasattr(self, 'coef_'):
            return self.coef_
        raise ValueError("Model not fitted")



# ---------- KEF 得分估计函数（内部标准化，还原梯度） ----------
def estimate_kef_gradients(X_train, X_query, lam=1e-4):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_query_scaled = scaler.transform(X_query).astype(np.float32)

    X_train_tensor = torch.tensor(X_train_scaled, device='cpu')
    X_query_tensor = torch.tensor(X_query_scaled, device='cpu')

    kernel = CurlFreeIMQ()
    estimator = Tikhonov(lam=lam, use_cg=True, kernel=kernel)
    estimator.fit(X_train_tensor)
    grad_tf = estimator.compute_gradients(X_query_tensor)

    grad = grad_tf.numpy()
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad = grad / (scaler.scale_ + 1e-8)
    return grad

# ---------- 数据生成（AR(1) 协方差） ----------
def ar1_cov(d, rho, min_eig=1e-6):
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov[i, j] = rho ** abs(i - j)
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_eig)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return cov

def generate_data(a, n, dom, sde, nsf, ar_rho=0.6):
    cov = ar1_cov(len(a), ar_rho)
    sample = np.random.multivariate_normal(np.zeros_like(a), cov, size=n)
    inp = sample @ a
    out = nsf[dom](inp)
    error = np.random.normal(0, sde, n)
    out += error
    return out, sample

def generate_unlabeled(a, n, dom, nsf, ar_rho=0.6):
    cov = ar1_cov(len(a), ar_rho)
    X = np.random.multivariate_normal(np.zeros_like(a), cov, size=n)
    return X

# ---------- 辅助函数 ----------
def st(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ht(data, threshold):
    return np.where(np.abs(data) > threshold, data, 0)

def sel_tun_lam(grids, n_folds, y, score, dom):
    kf = KFold(n_splits=n_folds, shuffle=True)
    dist = np.zeros((len(grids), n_folds))
    i = 0
    for train_idx, test_idx in kf.split(y):
        score_train, score_test = score[train_idx], score[test_idx]
        y_train, y_test = y[train_idx].reshape(-1,1), y[test_idx].reshape(-1,1)
        emp_p = np.mean(y_train * score_train, axis=0)
        test = np.mean(y_test * score_test, axis=0)
        for j, grid in enumerate(grids):
            if dom == 0:
                thres = st(emp_p, grid)
            else:
                thres = ht(emp_p, grid)
            dist[j, i] = np.mean((thres - test) ** 2)
        i += 1
    dist = np.mean(dist, axis=1)
    return grids[np.argmin(dist)]

def select_sign(ps, pt):
    idx = np.where((ps != 0) & (pt != 0))[0]
    if len(idx) == 0:
        return 1
    ps_sub = ps[idx]
    pt_sub = pt[idx]
    s_ps = np.sign(ps_sub)
    s_pt = np.sign(pt_sub)
    if np.sum(s_ps == s_pt) >= np.sum(-s_ps == s_pt):
        return 1
    else:
        return -1

def cross_validation(nfolds, gammas, input, output, estimator, base_estimator):
    kf = KFold(n_splits=nfolds, shuffle=True)
    dist = np.zeros((len(gammas), nfolds))
    i = 0
    for train_index, test_index in kf.split(output):
        input_train, input_test = input[train_index], input[test_index]
        output_train, output_test = output[train_index], output[test_index].reshape(-1,1)
        for j, gamma in enumerate(gammas):
            delta = st(estimator - base_estimator, gamma)
            sel_est = estimator - delta
            norm_est = np.linalg.norm(sel_est)
            if norm_est == 0:
                dist[j,i] = 1e10
                continue
            sel_est = sel_est / norm_est
            z_train = input_train @ sel_est
            z_test = input_test @ sel_est
            lr_model = LinearRegression().fit(z_train.reshape(-1,1), output_train.ravel())
            y_pred = lr_model.predict(z_test.reshape(-1,1)).ravel()
            outcome = mean_absolute_error(output_test.ravel(), y_pred)
            dist[j,i] = outcome
        i += 1
    dist = np.mean(dist, axis=1)
    print("  各 gamma 对应的平均 MAE:")
    for g, d in zip(gammas, dist):
        print(f"    gamma={g:.4f}, MAE={d:.6f}")
    best_gamma = gammas[np.argmin(dist)]
    print(f"  选择 gamma = {best_gamma:.4f}")
    return best_gamma, dist

# ---------- 评估器模型（固定投影神经网络） ----------
class ImprovedTNN(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

# ---------- 评估器模型（可微调方向 + MLP） ----------
class ImprovedSimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        proj_out = self.proj(x)
        out = self.mlp(proj_out)
        return out.squeeze()

# ---------- 评估器训练函数（固定投影） ----------
def train_improved_tnn(model, X, y, epochs=3000, lr=0.002, batch_size=256,
                       patience=500, val_ratio=0.2, device='cpu', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    X_mean, X_std = X.mean(), X.std()
    if X_std < 1e-8:
        X_std = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    n = len(X)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    X_train, X_val = X_norm[train_idx], X_norm[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        avg_train_loss = total_loss / len(train_idx)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        scheduler.step()

        if (epoch+1) % 200 == 0:
            print(f"TNN Epoch {epoch+1}/{epochs}, Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"TNN early stopping at epoch {epoch+1}, best val loss={best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(X_new):
        X_new_norm = (X_new - X_mean) / X_std
        with torch.no_grad():
            pred_norm = model(torch.tensor(X_new_norm, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()
        return pred_norm * y_std + y_mean
    return model, predict

def nn_estimator_fixed(proj_dir, X_train, y_train, X_test, epochs=3000, lr=0.002,
                       patience=500, hidden_dim=512, device='cpu', seed=42):
    if np.linalg.norm(proj_dir) == 0 or np.isnan(proj_dir).any():
        return np.zeros(X_test.shape[0])
    proj_dir = proj_dir / np.linalg.norm(proj_dir)
    z_train = (X_train @ proj_dir).reshape(-1, 1)
    z_test = (X_test @ proj_dir).reshape(-1, 1)
    model = ImprovedTNN(hidden_dim=hidden_dim, dropout=0.3)
    _, predict_fn = train_improved_tnn(model, z_train, y_train, epochs=epochs, lr=lr,
                                       batch_size=256, patience=patience, val_ratio=0.2,
                                       device=device, seed=seed)
    return predict_fn(z_test)

# ---------- 评估器训练函数（可微调方向） ----------
def train_improved_simplenn(model, X, y, X_test=None, epochs=3000, lr=0.002, batch_size=256,
                            patience=500, val_ratio=0.2, init_dir=None, device='cpu', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if init_dir is not None:
        with torch.no_grad():
            model.proj.weight.data = torch.tensor(init_dir.reshape(1, -1), dtype=torch.float32)
            model.proj.weight.data /= (model.proj.weight.data.norm() + 1e-8)

    model.to(device)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    n = len(X)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    X_train, X_val = X_norm[train_idx], X_norm[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        avg_train_loss = total_loss / len(train_idx)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        scheduler.step()

        if (epoch+1) % 200 == 0:
            print(f"SimpleNN Epoch {epoch+1}/{epochs}, Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"SimpleNN early stopping at epoch {epoch+1}, best val loss={best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_dir = model.proj.weight.data.cpu().numpy().flatten()
    if np.linalg.norm(final_dir) > 0:
        final_dir = final_dir / np.linalg.norm(final_dir)

    def predict(X_new):
        X_new_norm = (X_new - X_mean) / X_std
        with torch.no_grad():
            pred_norm = model(torch.tensor(X_new_norm, dtype=torch.float32).to(device)).cpu().numpy()
        return pred_norm * y_std + y_mean

    if X_test is not None:
        y_pred = predict(X_test)
        return model, y_pred, final_dir
    else:
        return model, None, final_dir

def nn_estimator_finetune(init_dir, X_train, y_train, X_test, epochs=3000, lr=0.002,
                          patience=100, hidden_dim=512, device='cpu', seed=42):
    model = ImprovedSimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=0.3)
    _, y_pred, final_dir = train_improved_simplenn(model, X_train, y_train, X_test,
                                                   epochs=epochs, lr=lr, batch_size=256,
                                                   patience=patience, val_ratio=0.2,
                                                   init_dir=init_dir, device=device, seed=seed)
    return y_pred, final_dir

# ---------- 一次实验 ----------
def run_single_setting(a0, source_dirs_true,
                       X_train_label, y_train_label,   # 用于交叉矩估计的有标签数据
                       X_train_all, y_train_all,       # 用于 Lasso 和 GLMtrans 的全量数据
                       X_test_label, y_test_label,
                       X_target_unlabel,               # 用于 KEF 的无标签数据
                       source_label_data, source_unlabel_data,
                       source_all_data,                 # 各源域的全量数据
                       ar_rho, sde, nsf, d, K, nn_hidden, nn_epochs, nn_lr,
                       grids_beta, grids_gamma, nfolds,
                       kef_lam=1e-5, rep_seed=0,
                       glmtrans_wrapper=None):
    
    # ---------- 目标方向估计（KEF）：只使用无标签数据 ----------
    if X_target_unlabel is None or len(X_target_unlabel) == 0:
        raise ValueError("目标域无标签数据为空，请提供无标签数据。")
    X_kef_target = X_target_unlabel
    print(f"  目标域训练 KEF 使用无标签样本: {X_kef_target.shape[0]}")

    score_target = estimate_kef_gradients(X_kef_target, X_train_label, lam=kef_lam)
    score_target = np.clip(score_target, np.quantile(score_target, 0.1), np.quantile(score_target, 0.9))
    lam = sel_tun_lam(grids_beta, min(5, len(y_train_label)//2), y_train_label.reshape(-1,1), score_target, 0)
    bas_a0 = np.mean(y_train_label.reshape(-1,1) * score_target, axis=0)
    bas_a0 = st(bas_a0, lam)
    if np.linalg.norm(bas_a0) == 0:
        bas_a0 = np.mean(y_train_label.reshape(-1,1) * score_target, axis=0)
    bas_a0 = bas_a0 / (np.linalg.norm(bas_a0) + 1e-8)

    # ---------- 源域方向估计（每个源域只估计一次，只使用无标签数据） ----------
    source_dirs_raw = []
    for i in range(K):
        if source_unlabel_data[i] is None or len(source_unlabel_data[i]) == 0:
            raise ValueError(f"源域 {i+1} 无标签数据为空，请提供无标签数据。")
        X_src_combined = source_unlabel_data[i]
        print(f"  源域 {i+1} 训练 KEF 使用无标签样本: {X_src_combined.shape[0]}")
        X_src_label, y_src_label = source_label_data[i]
        score_src = estimate_kef_gradients(X_src_combined, X_src_label, lam=kef_lam)
        score_src = np.clip(score_src, np.quantile(score_src, 0.1), np.quantile(score_src, 0.9))
        lam_src = sel_tun_lam(grids_beta, 5, y_src_label.reshape(-1,1), score_src, 1)
        a0i = np.mean(y_src_label.reshape(-1,1) * score_src, axis=0)
        a0i = ht(a0i, lam_src)
        if np.linalg.norm(a0i) == 0:
            a0i = np.mean(y_src_label.reshape(-1,1) * score_src, axis=0)
        a0i = a0i / (np.linalg.norm(a0i) + 1e-8)
        sign = select_sign(a0i, bas_a0)
        a0i = sign * a0i
        source_dirs_raw.append(a0i)

    # ---------- 收缩对齐 ----------
    source_dirs_final = []
    for a0i_raw in source_dirs_raw:
        best_gamma, _ = cross_validation(nfolds, grids_gamma, X_train_label, y_train_label,
                                         a0i_raw, bas_a0)
        delta = st(a0i_raw - bas_a0, best_gamma)
        a0i = a0i_raw - delta
        if np.linalg.norm(a0i) == 0:
            a0i = a0i_raw
        a0i = a0i / (np.linalg.norm(a0i) + 1e-8)
        a0i = select_sign(a0i, bas_a0) * a0i
        source_dirs_final.append(a0i)

    a_s_raw = np.mean(source_dirs_raw, axis=0)
    best_gamma_as, _ = cross_validation(nfolds, grids_gamma, X_train_label, y_train_label,
                                        a_s_raw, bas_a0)
    delta_as = st(a_s_raw - bas_a0, best_gamma_as)
    a_s = a_s_raw - delta_as
    if np.linalg.norm(a_s) == 0:
        a_s = a_s_raw
    a_s = a_s / (np.linalg.norm(a_s) + 1e-8)
    a_s = select_sign(a_s, bas_a0) * a_s

    if len(source_dirs_final) > 1:
        gamma_opt, _, _ = frank_wolfe_spherical_combination(np.array(source_dirs_final), bas_a0,
                                                           max_iterations=50000, tolerance=1e-6)
        gamma_opt = gamma_opt.reshape(-1,1)
        a0oc = np.mean(gamma_opt * np.array(source_dirs_final), axis=0)
        if np.linalg.norm(a0oc) == 0:
            a0oc = bas_a0
        a0oc = a0oc / (np.linalg.norm(a0oc) + 1e-8)
        a0oc = select_sign(a0oc, bas_a0) * a0oc
    else:
        a0oc = source_dirs_final[0].copy()

    # ---------- 评估函数 ----------
    def compute_metrics(dir_vec):
        if np.linalg.norm(dir_vec) == 0:
            return np.inf, np.inf, 180.0
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        if np.dot(dir_vec, a0) < 0:
            dir_vec = -dir_vec
        l2 = np.linalg.norm(dir_vec - a0)
        l1 = np.linalg.norm(dir_vec - a0, ord=1)
        angle = np.arccos(np.clip(np.dot(dir_vec, a0), -1, 1)) * 180 / np.pi
        return l2, l1, angle

    # ---------- Lasso（使用全量数据） ----------
    lasso = LassoCV(cv=5).fit(X_train_all, y_train_all)
    y_pred_lasso = lasso.predict(X_test_label)
    mse_lasso = mean_squared_error(y_test_label, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test_label, y_pred_lasso)
    r2_lasso = r2_score(y_test_label, y_pred_lasso)
    lasso_dir = lasso.coef_
    if np.linalg.norm(lasso_dir) > 0:
        lasso_dir = lasso_dir / np.linalg.norm(lasso_dir)

    # ---------- GLMtrans（使用全量数据，仅在 wrapper 存在时运行） ----------
    if glmtrans_wrapper is not None:
        try:
            X_sources = [src[0] for src in source_all_data]
            y_sources = [src[1] for src in source_all_data]
            glmtrans_wrapper.fit(X_train_all, y_train_all, X_source_list=X_sources, y_source_list=y_sources,
                                 method="transfer", transfer_source_id="all")
            y_pred_glm = glmtrans_wrapper.predict(X_test_label)
            mse_glm = mean_squared_error(y_test_label, y_pred_glm)
            mae_glm = mean_absolute_error(y_test_label, y_pred_glm)
            r2_glm = r2_score(y_test_label, y_pred_glm)
            coef_glm = glmtrans_wrapper.get_coefficients()[1:]  # 去掉截距项
            if np.linalg.norm(coef_glm) > 0:
                coef_glm = coef_glm / np.linalg.norm(coef_glm)
            sign_glm = select_sign(coef_glm, a0)
            glm_dir_aligned = sign_glm * coef_glm
            l2_glm = np.linalg.norm(glm_dir_aligned - a0)
            l1_glm = np.linalg.norm(glm_dir_aligned - a0, ord=1)
            angle_glm = np.arccos(np.clip(np.dot(glm_dir_aligned, a0), -1, 1)) * 180 / np.pi
        except Exception as e:
            print(f"GLMtrans failed: {e}")
            mse_glm = mae_glm = r2_glm = l2_glm = l1_glm = angle_glm = np.nan
            coef_glm = np.zeros(d)
    else:
        # 若未启用，设为 NaN
        mse_glm = mae_glm = r2_glm = l2_glm = l1_glm = angle_glm = np.nan
        coef_glm = np.zeros(d)

    # ---------- 随机方向（训练） ----------
    random_dir = np.random.randn(d)
    random_dir = random_dir / np.linalg.norm(random_dir)
    y_pred_rand_learn, final_rand = nn_estimator_finetune(random_dir, X_train_label, y_train_label, X_test_label,
                                                          epochs=nn_epochs, lr=nn_lr, patience=500,
                                                          hidden_dim=nn_hidden, device='cpu', seed=rep_seed+1000)
    mse_rand_learn = mean_squared_error(y_test_label, y_pred_rand_learn)
    mae_rand_learn = mean_absolute_error(y_test_label, y_pred_rand_learn)
    r2_rand_learn = r2_score(y_test_label, y_pred_rand_learn)

    # ---------- 固定神经网络 ----------
    y_pred_fixed_bas = nn_estimator_fixed(bas_a0, X_train_label, y_train_label, X_test_label,
                                          epochs=nn_epochs, lr=nn_lr, patience=50,
                                          hidden_dim=nn_hidden, device='cpu', seed=rep_seed+2000)
    mse_fixed_bas = mean_squared_error(y_test_label, y_pred_fixed_bas)
    mae_fixed_bas = mean_absolute_error(y_test_label, y_pred_fixed_bas)
    r2_fixed_bas = r2_score(y_test_label, y_pred_fixed_bas)

    y_pred_fixed_as = nn_estimator_fixed(a_s, X_train_label, y_train_label, X_test_label,
                                         epochs=nn_epochs, lr=nn_lr, patience=50,
                                         hidden_dim=nn_hidden, device='cpu', seed=rep_seed+3000)
    mse_fixed_as = mean_squared_error(y_test_label, y_pred_fixed_as)
    mae_fixed_as = mean_absolute_error(y_test_label, y_pred_fixed_as)
    r2_fixed_as = r2_score(y_test_label, y_pred_fixed_as)

    y_pred_fixed_oc = nn_estimator_fixed(a0oc, X_train_label, y_train_label, X_test_label,
                                         epochs=nn_epochs, lr=nn_lr, patience=50,
                                         hidden_dim=nn_hidden, device='cpu', seed=rep_seed+4000)
    mse_fixed_oc = mean_squared_error(y_test_label, y_pred_fixed_oc)
    mae_fixed_oc = mean_absolute_error(y_test_label, y_pred_fixed_oc)
    r2_fixed_oc = r2_score(y_test_label, y_pred_fixed_oc)

    # ---------- 收集结果 ----------
    results = {}
    for name, dir_vec, mse, mae, r2 in [
        ("lasso", lasso_dir, mse_lasso, mae_lasso, r2_lasso),
        ("glmtrans", coef_glm, mse_glm, mae_glm, r2_glm),
        ("rand_learn", final_rand, mse_rand_learn, mae_rand_learn, r2_rand_learn),
        ("fixed_bas", bas_a0, mse_fixed_bas, mae_fixed_bas, r2_fixed_bas),
        ("fixed_as", a_s, mse_fixed_as, mae_fixed_as, r2_fixed_as),
        ("fixed_oc", a0oc, mse_fixed_oc, mae_fixed_oc, r2_fixed_oc)
    ]:
        l2, l1, angle = compute_metrics(dir_vec)
        results[name] = [l2, l1, angle, mse, mae, r2]
    return results

# ---------- 主实验 ----------
def run_semi_supervised_simulation(args):
    d = args.dim
    s = args.sparsity
    K = args.n_sources
    sde = args.noise_std
    l, u = 0.9, 1
    ar_rho = args.rho
    perturb_rho = 0.999
    n_dif = 1
    rep = args.repeats
    nn_hidden = args.nn_hidden
    nn_epochs = args.nn_epochs
    nn_lr = args.nn_lr

    n_target_train = args.n_target_train
    n_target_test = args.n_label_target_test
    target_labeled_ratio = args.target_labeled_ratio

    n_source_train = args.n_source_train
    source_labeled_ratio = args.source_labeled_ratio

    kef_lam = args.kef_lam

    nsf = [lambda x: np.tanh(x)]
    nsf.append(lambda x: erf(x - 0.25))
    nsf.append(lambda x: np.sin(x - 0.25))
    nsf.append(lambda x: expit(x - 0.25))
    nsf.append(lambda x: np.tanh(x+0.25))
    nsf.append(lambda x: erf(x + 0.25))
    nsf.append(lambda x: np.sin(x + 0.25))
    nsf.append(lambda x: expit(x + 0.25))


    grids_beta = np.logspace(-4, 1, 20)
    grids_gamma = np.logspace(-4, 1, 20)
    nfolds = 5

    method_names = [
        "lasso", "glmtrans", "rand_learn", "fixed_bas", "fixed_as", "fixed_oc"
    ]

    interim_file = f"interim_semi_K{args.n_sources}_trainT{args.n_target_train}_labT{args.target_labeled_ratio:.2f}_trainS{args.n_source_train}_labS{args.source_labeled_ratio:.2f}_lam{args.kef_lam:.8f}_mix.pkl"

    if os.path.exists(interim_file):
        try:
            with open(interim_file, 'rb') as f:
                interim_data = pkl.load(f)
            results = interim_data['results']
            true_diff_metrics = interim_data['true_diff_metrics']
            start_rep = interim_data['start_rep']
            print(f"从中间文件恢复，已完成 {start_rep} 次重复")
        except Exception as e:
            print(f"加载中间文件失败: {e}，从头开始")
            results = {m: [] for m in method_names}
            true_diff_metrics = []
            start_rep = 0
    else:
        results = {m: [] for m in method_names}
        true_diff_metrics = []
        start_rep = 0

    if start_rep >= rep:
        print(f"已完成全部 {rep} 次重复，跳过实验")
    else:
        print(f"从第 {start_rep+1} 次重复开始继续实验")

    # ---- 根据开关决定是否加载 GLMtrans ----
    if args.use_glmtrans:
        glmtrans_wrapper = GLMTransWrapper()
        print("GLMtrans 已启用")
    else:
        glmtrans_wrapper = None
        print("GLMtrans 已禁用")

    for r in range(start_rep, rep):
        print(f"\n========== Replication {r+1}/{rep} ==========")
        np.random.seed(r)
        torch.manual_seed(r)

        a0 = gen_a0(d, s, l, u)

        source_dirs_true = []
        for _ in range(K):
            ak = gen_ak(a0, s, n_dif, d, perturb_rho)
            source_dirs_true.append(ak)

        src_l2, src_l1, src_angle = [], [], []
        for i, true_dir in enumerate(source_dirs_true):
            l2 = np.linalg.norm(true_dir - a0)
            l1 = np.linalg.norm(true_dir - a0, ord=1)
            angle = np.arccos(np.clip(np.dot(true_dir, a0), -1, 1)) * 180 / np.pi
            src_l2.append(l2)
            src_l1.append(l1)
            src_angle.append(angle)
            print(f"  源域 {i+1}: L2={l2:.4f}, L1={l1:.4f}, Angle={angle:.2f}°")
        true_diff_metrics.append((np.mean(src_l2), np.mean(src_l1), np.mean(src_angle)))

        # -------- 目标域数据 --------
        y_train_all, X_train_all = generate_data(a0, n_target_train, 0, sde, nsf, ar_rho)
        y_test_label, X_test_label = generate_data(a0, n_target_test, 0, sde, nsf, ar_rho)

        n_train_target = len(X_train_all)
        indices_target = np.random.permutation(n_train_target)
        n_labeled_target = int(n_train_target * target_labeled_ratio)
        labeled_idx_t = indices_target[:n_labeled_target]
        unlabeled_idx_t = indices_target[n_labeled_target:]
        X_labeled_target = X_train_all[labeled_idx_t]
        y_labeled_target = y_train_all[labeled_idx_t]
        X_unlabeled_target = X_train_all[unlabeled_idx_t]

        print(f"  目标域: 总训练样本 {n_train_target}, 有标签 {len(X_labeled_target)}, 无标签 {len(X_unlabeled_target)}")

        # -------- 源域数据 --------
        source_label_data = []
        source_unlabel_data = []
        source_all_data = []
        for kk in range(K):
            y_src_all, X_src_all = generate_data(source_dirs_true[kk], n_source_train, kk+1, sde, nsf, ar_rho)
            source_all_data.append((X_src_all, y_src_all))

            n_train_src = len(X_src_all)
            indices_src = np.random.permutation(n_train_src)
            n_labeled_src = int(n_train_src * source_labeled_ratio)
            labeled_idx_s = indices_src[:n_labeled_src]
            unlabeled_idx_s = indices_src[n_labeled_src:]
            X_labeled_src = X_src_all[labeled_idx_s]
            y_labeled_src = y_src_all[labeled_idx_s]
            X_unlabeled_src = X_src_all[unlabeled_idx_s]

            source_label_data.append((X_labeled_src, y_labeled_src))
            source_unlabel_data.append(X_unlabeled_src)

            print(f"  源域 {kk+1}: 总训练样本 {n_train_src}, 有标签 {len(X_labeled_src)}, 无标签 {len(X_unlabeled_src)}")

        # 调用 run_single_setting
        res = run_single_setting(
            a0, source_dirs_true,
            X_labeled_target, y_labeled_target,
            X_train_all, y_train_all,
            X_test_label, y_test_label,
            X_unlabeled_target,
            source_label_data, source_unlabel_data,
            source_all_data,
            ar_rho, sde, nsf, d, K, nn_hidden, nn_epochs, nn_lr,
            grids_beta, grids_gamma, nfolds,
            kef_lam=kef_lam,
            rep_seed=r,
            glmtrans_wrapper=glmtrans_wrapper
        )

        for name in method_names:
            if name in res:
                results[name].append(res[name])
            else:
                results[name].append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        print(f"\n  Rep {r+1} 详细结果:")
        for mname in method_names:
            if mname in res:
                l2, l1, ang, mse, mae, r2 = res[mname]
                print(f"    {mname}: L2={l2:.4f}, Angle={ang:.2f}°, MSE={mse:.4f}, R2={r2:.4f}")
            else:
                print(f"    {mname}: NaN")
        print("")

        interim_data = {
            'results': results,
            'true_diff_metrics': true_diff_metrics,
            'start_rep': r + 1
        }
        with open(interim_file, 'wb') as f:
            pkl.dump(interim_data, f)
        print(f"中间结果已保存至 {interim_file} (已运行 {r+1} 次重复)")

        print(f"Rep {r+1} done. Fixed_oc R2: {res['fixed_oc'][5]:.4f}")

    # 保存最终结果
    output_file = f"simulation_K{args.n_sources}_trainT{args.n_target_train}_labT{args.target_labeled_ratio:.2f}_trainS{args.n_source_train}_labS{args.source_labeled_ratio:.2f}_lam{args.kef_lam:.8f}_KEF_mix.pkl"
    with open(output_file, "wb") as f:
        pkl.dump({
            "results": results,
            "true_diff_metrics": true_diff_metrics
        }, f)
    print(f"\nSimulation results saved to {output_file}")

    if os.path.exists(interim_file):
        os.remove(interim_file)
        print(f"中间文件 {interim_file} 已删除")

    def print_stats(name, arr):
        if arr.shape[0] == 0 or np.all(np.isnan(arr)):
            print(f"{name:15} | All NaN")
            return
        metrics = ["L2", "L1", "Angle(deg)", "MSE", "MAE", "R2"]
        for idx, met in enumerate(metrics):
            vals = arr[:, idx]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            median = np.nanmedian(vals)
            q25 = np.nanpercentile(vals, 25)
            q75 = np.nanpercentile(vals, 75)
            if met == "Angle(deg)":
                print(f"{name:15} | {met:10} | Mean={mean:.1f}±{std:.1f} | Med={median:.1f} | [25%,75%]=[{q25:.1f},{q75:.1f}]")
            else:
                print(f"{name:15} | {met:10} | Mean={mean:.4f}±{std:.4f} | Med={median:.4f} | [25%,75%]=[{q25:.4f},{q75:.4f}]")

    print("\n========== Results ==========")
    for name in method_names:
        arr = np.array(results[name])
        if arr.shape[0] == 0 or np.all(np.isnan(arr)):
            print(f"{name:15} | All NaN")
        else:
            print_stats(name, arr)
    print("\n========== Source-to-Target True Direction Differences ==========")
    true_arr = np.array(true_diff_metrics)
    l2_mean, l2_std = true_arr[:,0].mean(), true_arr[:,0].std()
    l1_mean, l1_std = true_arr[:,1].mean(), true_arr[:,1].std()
    ang_mean, ang_std = true_arr[:,2].mean(), true_arr[:,2].std()
    print(f"Average over {rep} replications: L2={l2_mean:.4f}±{l2_std:.4f}, L1={l1_mean:.4f}±{l1_std:.4f}, Angle={ang_mean:.2f}°±{ang_std:.2f}°")

# 辅助函数
def gen_a0(d, s, l, u):
    sign = np.random.binomial(1, 0.5, s) * 2 - 1
    a0 = np.zeros(d)
    nz = np.random.uniform(l, u, s)
    nz = sign * nz
    a0[:s] = nz
    a0 = a0 / np.linalg.norm(a0)
    return a0

def gen_ak(a0, s, n, d, rho):
    ak = copy.deepcopy(a0)
    id1 = np.random.randint(0, s, n)
    id2 = np.random.randint(s, d, n)
    ak[id2] = ak[id1]
    ak[id1] = rho * ak[id1]
    ak = ak / np.linalg.norm(ak)
    return ak

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--sparsity", type=int, default=30)
    parser.add_argument("--n_sources", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=1/4)
    parser.add_argument("--rho", type=float, default=1/8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--nn_hidden", type=int, default=64)
    parser.add_argument("--nn_epochs", type=int, default=5000)
    parser.add_argument("--nn_lr", type=float, default=0.002)

    # 目标域数据参数
    parser.add_argument("--n_target_train", type=int, default=500,
                        help="目标域训练集总样本数")
    parser.add_argument("--target_labeled_ratio", type=float, default=0.5,
                        help="目标域训练集中有标签比例")
    parser.add_argument("--n_label_target_test", type=int, default=1500,
                        help="目标域测试集样本数")

    # 源域数据参数
    parser.add_argument("--n_source_train", type=int, default=6000,
                        help="每个源域训练集总样本数")
    parser.add_argument("--source_labeled_ratio", type=float, default=0.5,
                        help="每个源域训练集中有标签比例")

    # KEF 参数
    parser.add_argument("--kef_lam", type=float, default=1e-5,
                        help="KEF 正则化参数")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径")
    # GLMtrans 开关
    parser.add_argument("--use_glmtrans", action="store_true", default=False,
                        help="是否启用 GLMtrans（需要 R 环境）")
    args = parser.parse_args()

    if args.log_file is None or args.log_file == "simulation.log":
        log_filename = f"simulation_K{args.n_sources}_trainT{args.n_target_train}_labT{args.target_labeled_ratio:.2f}_trainS{args.n_source_train}_labS{args.source_labeled_ratio:.2f}_KEF_mix.log"
    else:
        log_filename = args.log_file

    class Tee:
        def __init__(self, filename):
            self.file = open(filename, 'w', encoding='utf-8', buffering=1)
            self.stdout = sys.stdout
        def write(self, message):
            self.stdout.write(message)
            self.stdout.flush()
            self.file.write(message)
            self.file.flush()
        def flush(self):
            self.stdout.flush()
            self.file.flush()
        def close(self):
            self.file.close()

    tee = Tee(log_filename)
    sys.stdout = tee
    print(f"日志文件: {os.path.abspath(log_filename)}")
    print(f"KEF 正则化参数: {args.kef_lam}")
    print(f"目标域: 训练集总样本 {args.n_target_train}, 有标签比例 {args.target_labeled_ratio}")
    print(f"源域: 训练集总样本 {args.n_source_train}, 有标签比例 {args.source_labeled_ratio}")
    print(f"GLMtrans 启用: {args.use_glmtrans}")
    print("="*80)

    try:
        run_semi_supervised_simulation(args)
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = tee.stdout
        tee.close()
