#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单池迁移实验（puma8NH 真实机器人动力学，8192 样本，仅用 R² 评价）。
  · 数据使用【原始尺度】(未预标准化)；域划分基于原始池(seed固定一次切好)。
  · 各方法“内部标准化”在【各自域内部】完成：目标域/每个源域各自用本域全部样本
    拟合 StandardScaler(mean/std)，仅作用于 X，y 保留原始量纲。
  · repeat 期间域划分固定，仅重抽标签与目标测试。


设定（单池切分，方案 A）：
  · 把 3 份切片合并成一个数据池，按 seed 固定切一次：
        - 源域 = n_source_domains × n_source_samples 个样本
        - 剩余样本全部作为目标域
  · 目标域内再按 n_label_target_test 划分测试，训练内按 target_labeled_ratio 划分有/无标签
  · 每次 repeat 只在源/目标域内重抽标签与目标测试，域划分固定（主流做法）
  · 源域内部按 source_labeled_ratio 【随机划分】有/无标签，
    选域阶段与 transfer 阶段共用同一次划分
  · 评价：真实数据无真实 index，去掉 L2/L1/夹角，仅保留 R²

流程：选域(可选,双重置换) -> Transfer(仅用选中源域) -> R²
"""
import os, sys, copy
import numpy as np

# ---- 环境探测 ----
try:
    print("[ENV] python:", sys.executable)
    print("[ENV] numpy :", np.__version__, "@", np.__file__)
except Exception as _e:
    print("[ENV] numpy version 读取失败:", repr(_e))

# ---- numpy 2.x -> numpy.core 兜底 ----
if not hasattr(np, 'core'):
    import numpy._core as _nc
    np.core = _nc
    _sys = sys
    _sys.modules['numpy.core'] = _nc
    for _sub in ('numeric', 'umath', 'multiarray', 'fromnumeric', 'arrayprint',
                 'getlimits', 'shape_base', 'ma', 'records', '_multiarray_umath',
                 '_internal', 'function_base'):
        if hasattr(_nc, _sub):
            _sys.modules[f'numpy.core.{_sub}'] = getattr(_nc, _sub)

# ---- numpy 旧别名兜底 ----
for _n, _v in [
    ('long', np.int64), ('ulong', np.uint64),
    ('int', np.int64), ('float', np.float64), ('bool', bool),
    ('complex', np.complex128), ('object', object),
    ('int_', np.int64), ('float_', np.float64), ('complex_', np.complex128),
    ('bool_', bool), ('string_', np.bytes_), ('unicode_', np.str_),
]:
    if not hasattr(np, _n):
        setattr(np, _n, _v)

import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from kscore.estimators import Tikhonov
    from kscore.kernels import CurlFreeIMQ
except ImportError:
    raise ImportError("请安装 numethod 包 (pip install numethod)，KEF 需要 kscore")

# Frank-Wolfe 组合（有真实模块则用，否则退化为均等权重的占位实现，并打印警告）
try:
    from Frank_Wolfe_Algorithm import frank_wolfe_spherical_combination
except ImportError:
    print("[警告] Frank_Wolfe_Algorithm 未找到，fixed_ow 将退化为均等权重组合")
    def frank_wolfe_spherical_combination(vectors, direction, max_iterations=50000,
                                          tolerance=1e-6):
        n = len(vectors)
        return np.ones((n, 1)) / n

# =========================================================================
# 基础辅助
# =========================================================================

def st(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ht(data, threshold):
    return np.where(np.abs(data) > threshold, data, 0)

def select_sign(ps, pt):
    idx = np.where((ps != 0) & (pt != 0))[0]
    if len(idx) == 0:
        return 1
    ps_sub = ps[idx]; pt_sub = pt[idx]
    s_ps = np.sign(ps_sub); s_pt = np.sign(pt_sub)
    if np.sum(s_ps == s_pt) >= np.sum(-s_ps == s_pt):
        return 1
    else:
        return -1

def sel_tun_lam(grids, n_folds, y, score, dom):
    kf = KFold(n_splits=n_folds, shuffle=True)
    dist = np.zeros((len(grids), n_folds)); i = 0
    for train_idx, test_idx in kf.split(y):
        score_train, score_test = score[train_idx], score[test_idx]
        y_train, y_test = y[train_idx].reshape(-1, 1), y[test_idx].reshape(-1, 1)
        emp_p = np.mean(y_train * score_train, axis=0)
        test = np.mean(y_test * score_test, axis=0)
        for j, grid in enumerate(grids):
            thres = st(emp_p, grid) if dom == 0 else ht(emp_p, grid)
            dist[j, i] = np.mean((thres - test) ** 2)
        i += 1
    dist = np.mean(dist, axis=1)
    return grids[np.argmin(dist)]

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

def estimate_direction_from_score(score, X_label, y_label, grids_beta, dom):
    """给定预先计算好的得分矩阵，估计方向。
    dom=0：目标域（软阈值），dom=1：源域（硬阈值）。
    —— 与 rank_sel_new 原版一致：无论是否零向量，最终都归一化。——"""
    score = np.clip(score, np.quantile(score, 0.1), np.quantile(score, 0.9))
    if dom == 0:
        lam_sel = sel_tun_lam(grids_beta, min(5, len(y_label) // 2),
                              y_label.reshape(-1, 1), score, 0)
        raw = np.mean(y_label.reshape(-1, 1) * score, axis=0)
        dir_vec = st(raw, lam_sel)
    else:
        lam_sel = sel_tun_lam(grids_beta, 5, y_label.reshape(-1, 1), score, 1)
        raw = np.mean(y_label.reshape(-1, 1) * score, axis=0)
        dir_vec = ht(raw, lam_sel)
    # 关键修复 A1：零向量兜底 + 统一归一化（归一化必须在 if 外）
    if np.linalg.norm(dir_vec) == 0:
        dir_vec = np.mean(y_label.reshape(-1, 1) * score, axis=0)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    return dir_vec

def estimate_base_direction_kef_semi(X_unlabel, X_label, y_label, lam=1e-4, grids_beta=None):
    if grids_beta is None:
        grids_beta = np.logspace(-4, 1, 20)
    score = estimate_kef_gradients(X_unlabel, X_label, lam=lam)
    return estimate_direction_from_score(score, X_label, y_label, grids_beta, dom=0)

def _dir_dist(a, b, norm="l1"):
    """方向差异距离。norm: l1 / l2 / inf(max)"""
    d = a - b
    if norm == "l1":
        return np.sum(np.abs(d))
    elif norm == "l2":
        return np.sqrt(np.sum(d * d))
    else:
        return np.max(np.abs(d))

def sparse_topk(a, k):
    """只保留幅度最大的 k 个分量, 其余置0(方向稀疏化)。
    k<=0 或 k>=len 则原样返回。关键修复 A5：稀疏化后保持单位范数。"""
    if k is None or k <= 0 or k >= len(a):
        return np.array(a, dtype=float)
    a2 = np.zeros_like(a)
    idx = np.argsort(np.abs(a))[::-1][:k]
    a2[idx] = a[idx]
    nrm = np.linalg.norm(a2)
    if nrm > 0:
        a2 = a2 / nrm
    return a2

# =========================================================================
# 选域：双重置换检验（关键修复 A2 — 真正使用 dist_norm）
# =========================================================================

def adaptive_selection_by_permutation_double_fast(
        X_sources_unlabel, X_sources_label, y_sources_label,
        X_target_unlabel, X_target_label, y_target_label,
        target_dir, lam=1e-4, grids_beta=None,
        n_perm=200, alpha=0.05, fdr_control=True, dist_norm="l1", sparse_k=0):
    """双重置换检验选择源域：同时置换源域与目标域标签生成零分布。
    距离范数由 dist_norm 控制（l1 / l2 / inf）。"""
    if grids_beta is None:
        grids_beta = np.logspace(-4, 1, 20)
    if sparse_k and sparse_k > 0:
        target_dir = sparse_topk(target_dir, sparse_k)
    K = len(X_sources_label)
    p_values = np.ones(K)
    obs_dist = np.zeros(K)

    # 预计算得分矩阵
    source_scores = []
    for i in range(K):
        score = estimate_kef_gradients(X_sources_unlabel[i], X_sources_label[i], lam=lam)
        source_scores.append(score)
    target_score = estimate_kef_gradients(X_target_unlabel, X_target_label, lam=lam)
    n_tgt = len(y_target_label)

    for i in range(K):
        X_label_src = X_sources_label[i]; y_label_src = y_sources_label[i]
        score_src = source_scores[i]
        a_obs_src = estimate_direction_from_score(score_src, X_label_src, y_label_src,
                                                  grids_beta, dom=1)
        if sparse_k and sparse_k > 0:
            a_obs_src = sparse_topk(a_obs_src, sparse_k)
        if np.dot(a_obs_src, target_dir) < 0:
            a_obs_src = -a_obs_src
        obs_dist[i] = _dir_dist(a_obs_src, target_dir, dist_norm)

        null_dists = []
        for _ in range(n_perm):
            y_perm_src = y_label_src[np.random.permutation(len(y_label_src))]
            a_perm_src = estimate_direction_from_score(score_src, X_label_src, y_perm_src,
                                                       grids_beta, dom=1)
            y_perm_tgt = y_target_label[np.random.permutation(n_tgt)]
            a_perm_tgt = estimate_direction_from_score(target_score, X_target_label,
                                                       y_perm_tgt, grids_beta, dom=0)
            if np.dot(a_perm_src, target_dir) < 0:
                a_perm_src = -a_perm_src
            if np.dot(a_perm_tgt, target_dir) < 0:
                a_perm_tgt = -a_perm_tgt
            d_perm = _dir_dist(a_perm_src, a_perm_tgt, dist_norm)
            null_dists.append(d_perm)
        p_values[i] = np.mean(np.array(null_dists) <= obs_dist[i])

    if fdr_control:
        from statsmodels.stats.multitest import multipletests
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        selected = np.where(reject)[0].tolist()
    else:
        selected = np.where(p_values < alpha)[0].tolist()
    return selected, p_values, obs_dist

def cross_validation(nfolds, gammas, input, output, estimator, base_estimator):
    kf = KFold(n_splits=nfolds, shuffle=True)
    dist = np.zeros((len(gammas), nfolds)); i = 0
    for train_index, test_index in kf.split(output):
        input_train, input_test = input[train_index], input[test_index]
        output_train, output_test = output[train_index], output[test_index].reshape(-1, 1)
        for j, gamma in enumerate(gammas):
            delta = st(estimator - base_estimator, gamma)
            sel_est = estimator - delta
            norm_est = np.linalg.norm(sel_est)
            if norm_est == 0:
                dist[j, i] = 1e10
                continue
            sel_est = sel_est / norm_est
            z_train = input_train @ sel_est; z_test = input_test @ sel_est
            lr_model = LinearRegression().fit(z_train.reshape(-1, 1), output_train.ravel())
            y_pred = lr_model.predict(z_test.reshape(-1, 1)).ravel()
            dist[j, i] = mean_absolute_error(output_test.ravel(), y_pred)
        i += 1
    dist = np.mean(dist, axis=1)
    return gammas[np.argmin(dist)], dist

# =========================================================================
# NN 评估器
# =========================================================================

class ImprovedTNN(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.net(x).squeeze()

class ImprovedSimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.mlp(self.proj(x)).squeeze()

def train_improved_tnn(model, X, y, epochs=3000, lr=0.002, batch_size=256,
                       patience=500, val_ratio=0.2, device='cpu', seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    X_mean, X_std = X.mean(), X.std()
    if X_std < 1e-8: X_std = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std < 1e-8: y_std = 1.0
    y_norm = (y - y_mean) / y_std
    n = len(X)
    indices = np.random.permutation(n); n_val = int(n * val_ratio)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    X_train, X_val = X_norm[train_idx], X_norm[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]
    td = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                       torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(td, batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.HuberLoss(delta=0.5)
    best_val, patience_counter, best_state = float('inf'), 0, None
    for e in range(epochs):
        model.train(); tl = 0.0
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step(); tl += loss.item() * bx.size(0)
        model.eval()
        with torch.no_grad():
            vl = crit(model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)),
                      torch.tensor(y_val, dtype=torch.float32)).item()
        sched.step()
        if vl < best_val - 1e-4:
            best_val = vl; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    def predict(X_new):
        pred = model(torch.tensor((X_new - X_mean) / X_std,
                                  dtype=torch.float32).unsqueeze(1)).detach().numpy()
        return pred * y_std + y_mean
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

def train_improved_simplenn(model, X, y, X_test=None, epochs=3000, lr=0.002, batch_size=256,
                            patience=500, val_ratio=0.2, init_dir=None, device='cpu', seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    if init_dir is not None:
        with torch.no_grad():
            model.proj.weight.data = torch.tensor(init_dir.reshape(1, -1), dtype=torch.float32)
            model.proj.weight.data /= (model.proj.weight.data.norm() + 1e-8)
    model.to(device)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std < 1e-8: y_std = 1.0
    y_norm = (y - y_mean) / y_std
    n = len(X)
    indices = np.random.permutation(n); n_val = int(n * val_ratio)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    X_train, X_val = X_norm[train_idx], X_norm[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]
    td = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(td, batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.HuberLoss(delta=0.5)
    best_val, patience_counter, best_state = float('inf'), 0, None
    for e in range(epochs):
        model.train(); tl = 0.0
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step(); tl += loss.item() * bx.size(0)
        model.eval()
        with torch.no_grad():
            vl = crit(model(torch.tensor(X_val, dtype=torch.float32)),
                      torch.tensor(y_val, dtype=torch.float32)).item()
        sched.step()
        if vl < best_val - 1e-4:
            best_val = vl; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_dir = model.proj.weight.data.cpu().numpy().flatten()
    if np.linalg.norm(final_dir) > 0:
        final_dir = final_dir / np.linalg.norm(final_dir)
    def predict(X_new):
        pred = model(torch.tensor((X_new - X_mean) / X_std,
                                  dtype=torch.float32)).detach().numpy()
        return pred * y_std + y_mean
    if X_test is not None:
        return model, predict(X_test), final_dir
    return model, None, final_dir

def nn_estimator_finetune(init_dir, X_train, y_train, X_test, epochs=3000, lr=0.002,
                          patience=100, hidden_dim=512, device='cpu', seed=42):
    model = ImprovedSimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=0.3)
    _, y_pred, final_dir = train_improved_simplenn(model, X_train, y_train, X_test,
                                                   epochs=epochs, lr=lr, batch_size=256,
                                                   patience=patience, val_ratio=0.2,
                                                   init_dir=init_dir, device=device, seed=seed)
    return y_pred, final_dir

# =========================================================================
# 数据读取
# =========================================================================

SINGLE_FILE = "puma8NH_singleindex_multidomain.npz"
_ALL_CACHE = None

# 单文件里 X_1/X_2/X_3 三份切片，合并起来 = 整个数据池(8192)
CITIES = ["1", "2", "3"]
TARGET_NAMES = ["1"]

def load_domain(name):
    global _ALL_CACHE
    if _ALL_CACHE is None:
        _ALL_CACHE = np.load(SINGLE_FILE, allow_pickle=True)
    return _ALL_CACHE[f"X_{name}"], _ALL_CACHE[f"y_{name}"]


def build_pool_split(args, seed):
    """把整个数据(合并所有域)当作一个池，固定切 K 个源域 + 目标域=剩余全部样本。
    - K = args.n_source_domains, 每源域 = args.n_source_samples 个样本。
    - 剩余样本全部作为目标域（run_target_once 内再划训练/测试、训练内划标签）。
    - 若 noise_cols>0：为池追加独立标准高斯噪声列(制造高维稀疏冗余特征)。"""
    d = _ALL_CACHE if _ALL_CACHE is not None else np.load(SINGLE_FILE, allow_pickle=True)
    X = np.asarray(d['X'], dtype=float); y = np.asarray(d['y'], dtype=float)
    rng = np.random.RandomState(seed)
    nc = int(getattr(args, "noise_cols", 0))
    if nc > 0:
        noise = rng.randn(len(X), nc)              # 独立标准高斯
        noise = noise - noise.mean(0)
        noise = noise / (noise.std(0) + 1e-9)
        X = np.concatenate([X, noise], axis=1)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    K = int(args.n_source_domains); ns = int(args.n_source_samples)
    need = K * ns
    if need >= len(X):
        raise SystemExit(f"源域样本数 K*ns={need} >= 总样本{len(X)}, 需给目标域留样本")
    src = {}
    for k in range(K):
        a, b = k * ns, (k + 1) * ns
        src[f"s{k+1}"] = (X[a:b], y[a:b])
    return src, (X[need:], y[need:])

def split_domain(X, y, n_train, n_test, labeled_ratio=0.5):
    """非时序：随机抽测试 n_test，再从剩余随机抽训练 n_train。
    内部按 labeled_ratio 拆 labeled/unlabeled。"""
    n = len(X)
    need = n_train + n_test
    if need > n:
        raise ValueError(f"n_target_samples({n_train}) + n_label_target_test({n_test}) "
                         f"> 该域总样本({n})，请调小")
    idx = np.random.permutation(n)
    test_idx = idx[:n_test]
    rem = idx[n_test:n_test + n_train]
    n_lab = int(n_train * labeled_ratio)
    lab_idx, unl_idx = rem[:n_lab], rem[n_lab:]
    return (X[lab_idx], y[lab_idx]), (X[unl_idx], y[unl_idx]), (X[test_idx], y[test_idx])

def _split_source(Xs, ys, labeled_ratio):
    """单个源域【随机】划分有标签/无标签索引。"""
    n = len(Xs)
    idx = np.random.permutation(n)
    nl = int(n * labeled_ratio)
    return idx[:nl], idx[nl:]

# =========================================================================
# 单次目标域实验
# =========================================================================

def run_target_once(target_name, sources, args, rep_seed, base_target=None):
    verbose = not getattr(args, "quiet", False)
    def vprint(*a):
        if verbose:
            print(*a)

    Xt, yt = load_domain(target_name) if base_target is None else base_target
    n_train = args.n_target_samples
    if n_train <= 0:                      # 0 = 目标域训练样本用剩余全部
        n_train = len(Xt) - args.n_label_target_test
    (X_lab, y_lab), (X_unl, y_unl), (X_test, y_test) = split_domain(
        Xt, yt, n_train, args.n_label_target_test,
        args.target_labeled_ratio)
    # 目标域【内部】标准化(per-domain)：用本域全部样本 fit scaler，y 保留原始量纲
    _sc_t = StandardScaler().fit(Xt)
    X_lab = _sc_t.transform(X_lab)
    X_unl = _sc_t.transform(X_unl)
    X_test = _sc_t.transform(X_test)
    grids_beta = np.logspace(-4, 1, 20)
    grids_gamma = np.logspace(-4, 1, 20)
    vprint(f"    [目标 {target_name}] 有标签={len(X_lab)}, 无标签={len(X_unl)}, 测试={len(X_test)}")

    # ---- 关键修复 A4：源域在本次 repeat 只划分一次，选域与 transfer 共用同一划分 ----
    src_split = {}
    for gname, (Xs, ys) in sources.items():
        lab_idx, unl_idx = _split_source(Xs, ys, args.source_labeled_ratio)
        # 每个源域【内部】标准化(per-domain)：用该域全部样本 fit scaler，y 不标准化
        _sc_s = StandardScaler().fit(Xs)
        src_split[gname] = {
            "lab": (_sc_s.transform(Xs[lab_idx]), ys[lab_idx]),
            "unl": _sc_s.transform(Xs[unl_idx]),
            "n_total": len(Xs),
        }

    # ---- 阶段 0：目标方向 KEF（只用目标无标签）----
    target_dir = estimate_base_direction_kef_semi(X_unl, X_lab, y_lab,
                                                  lam=args.kef_lam, grids_beta=grids_beta)
    # 符号对齐到 y 正相关；归一化由 estimate_direction_from_score 保证
    if np.dot(X_lab @ target_dir, y_lab - y_lab.mean()) < 0:
        target_dir = -target_dir
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
    vprint(f"    [目标方向] ||target_dir||={np.linalg.norm(target_dir):.4f}, "
           f"非零={int(np.count_nonzero(target_dir))}")

    # ---- 阶段 1：选域（可选，双重置换检验）----
    selected = list(range(len(sources)))
    use_names = list(sources.keys())
    if args.use_selection:
        src_unl = [src_split[g]["unl"] for g in sources.keys()]
        src_lab = [src_split[g]["lab"] for g in sources.keys()]
        selected, pvals, obs = adaptive_selection_by_permutation_double_fast(
            src_unl, [l[0] for l in src_lab], [l[1] for l in src_lab],
            X_unl, X_lab, y_lab, target_dir,
            lam=args.kef_lam, grids_beta=grids_beta,
            n_perm=args.n_perm, alpha=args.alpha, fdr_control=args.fdr_control,
            dist_norm=args.dist_norm, sparse_k=args.sparse_k)
        names = list(sources.keys())
        use_names = [names[i] for i in selected]
        vprint(f"    [选域] p值={np.round(pvals, 4)}, 距离({args.dist_norm})={np.round(obs, 4)}, "
               f"选中={use_names}")
    else:
        vprint(f"    [选域] 已关闭，使用全部源域={use_names}")

    # ---- 阶段 2：Transfer（KEF + 收缩 + Frank-Wolfe）----
    # 关键修复 B1：dirs_raw（收缩前）用于 a_s 求均值，与 sim_tanh_new 一致
    dirs_raw = []
    dirs_final = []
    for gname in use_names:
        X_sl, y_sl = src_split[gname]["lab"]
        X_su = src_split[gname]["unl"]
        vprint(f"      [源域 {gname}] 抽取={src_split[gname]['n_total']}, "
               f"有标签={len(X_sl)}, 无标签={len(X_su)}")
        score_src = estimate_kef_gradients(X_su, X_sl, lam=args.kef_lam)
        a0i_raw = estimate_direction_from_score(score_src, X_sl, y_sl, grids_beta, dom=1)
        a0i_raw = select_sign(a0i_raw, target_dir) * a0i_raw
        dirs_raw.append(a0i_raw)

        bg, _ = cross_validation(5, grids_gamma, X_lab, y_lab, a0i_raw, target_dir)
        a0i = a0i_raw - st(a0i_raw - target_dir, bg)
        if np.linalg.norm(a0i) == 0:
            a0i = a0i_raw.copy()
        else:
            a0i = a0i / np.linalg.norm(a0i)
        a0i = select_sign(a0i, target_dir) * a0i
        dirs_final.append(a0i)

    if len(dirs_final) == 0:
        print("  [选域为空] 只用目标域(不迁移)（fixed_mean/fixed_ow 用目标方向）")
        a_s = target_dir.copy()
        a_oc = target_dir.copy()
    else:
        # 平均方向 a_s：与 sim_tanh_new 一致，用未收缩的 raw 求均值
        a_s_raw = np.mean(dirs_raw, axis=0)
        bg, _ = cross_validation(5, grids_gamma, X_lab, y_lab, a_s_raw, target_dir)
        a_s = a_s_raw - st(a_s_raw - target_dir, bg)
        if np.linalg.norm(a_s) == 0:
            a_s = a_s_raw
        a_s = a_s / (np.linalg.norm(a_s) + 1e-8)
        a_s = select_sign(a_s, target_dir) * a_s

        # Frank-Wolfe 组合方向 a_oc：用收缩后的 dirs_final
        if len(dirs_final) > 1:
            gamma_opt, _, _ = frank_wolfe_spherical_combination(
                np.array(dirs_final), target_dir, max_iterations=50000, tolerance=1e-6)
            a_oc = np.mean(gamma_opt.reshape(-1, 1) * np.array(dirs_final), axis=0)
            if np.linalg.norm(a_oc) == 0:
                a_oc = target_dir
            a_oc = a_oc / (np.linalg.norm(a_oc) + 1e-8)
            a_oc = select_sign(a_oc, target_dir) * a_oc
        else:
            a_oc = dirs_final[0].copy()

    vprint(f"    [Transfer] 源域方向数={len(dirs_final)}, "
           f"||a_s||={np.linalg.norm(a_s):.4f}, "
           f"||a_oc||={np.linalg.norm(a_oc):.4f}, "
           f"||target_dir||={np.linalg.norm(target_dir):.4f}")

    # ---- 阶段 3：评估（仅 R²）----
    def nn_r2(dirv, seed):
        yp = nn_estimator_fixed(dirv, X_lab, y_lab, X_test,
                                epochs=args.nn_epochs, lr=args.nn_lr, patience=200,
                                hidden_dim=args.nn_hidden, seed=seed)
        return float(r2_score(y_test, yp))

    results = {}
    X_all = np.concatenate([X_lab, X_unl]); y_all = np.concatenate([y_lab, y_unl])
    lasso = LassoCV(cv=min(5, len(X_all))).fit(X_all, y_all)
    results["lasso"] = float(r2_score(y_test, lasso.predict(X_test)))
    # lasso 仅用有标签样本(不借用无标签)更公平; 原 lasso 用 X_all 含无标签真实y属作弊
    lasso_lab = LassoCV(cv=min(5, len(X_lab))).fit(X_lab, y_lab)
    results["lasso_lab"] = float(r2_score(y_test, lasso_lab.predict(X_test)))

    rd = np.random.randn(X_lab.shape[1]); rd /= np.linalg.norm(rd)
    yp_rl, _ = nn_estimator_finetune(rd, X_lab, y_lab, X_test,
                                     epochs=args.nn_epochs, lr=args.nn_lr, patience=200,
                                     hidden_dim=args.nn_hidden, seed=rep_seed + 1000)
    results["rand_nn"] = float(r2_score(y_test, yp_rl))
    results["fixed_target"] = nn_r2(target_dir, rep_seed + 2000)
    results["fixed_mean"] = nn_r2(a_s, rep_seed + 3000)
    results["fixed_ow"] = nn_r2(a_oc, rep_seed + 4000)

    def dir_sanity(dirv, tag):
        ztr = X_lab @ dirv; ze = X_test @ dirv
        c = float(np.corrcoef(ztr, y_lab)[0, 1])
        lin = float(r2_score(y_test, LinearRegression().fit(ztr.reshape(-1, 1), y_lab)
                             .predict(ze.reshape(-1, 1))))
        vprint(f"    [sanity {tag:6s}] corr(z,y)={c:+.3f}  固定方向+线性R2={lin:+.4f}")
    if not getattr(args, "quiet", False):
        dir_sanity(target_dir, "target")
        dir_sanity(a_s, "mean"); dir_sanity(a_oc, "ow")

    vprint("    [评估R2] " + ", ".join(f"{m}={v:.4f}" for m, v in results.items()))
    vprint("    [使用源域] " + (", ".join(use_names) if use_names else "无(退化为不迁移)"))
    return results, use_names

# =========================================================================
# 主实验
# =========================================================================

def run_experiment(args):
    import torch
    global SINGLE_FILE, _ALL_CACHE, TARGET_NAMES
    tgt = args.target_city
    if tgt not in CITIES:
        raise SystemExit(f"未知目标域 {tgt}, 可选 {CITIES}")
    TARGET_NAMES = [tgt]
    if args.single_file:
        SINGLE_FILE = args.single_file
    _ALL_CACHE = None

    # ---- 数据读取校验（单文件模式） ----
    d_all = np.load(SINGLE_FILE, allow_pickle=True)
    _ALL_CACHE = d_all
    Xall = np.asarray(d_all['X'], dtype=float); yall = np.asarray(d_all['y'], dtype=float)
    pdim = Xall.shape[1]
    p_ok = Xall.ndim == 2 and Xall.shape[0] == len(yall)
    print(f"\n=== 数据读取校验 (单文件 {SINGLE_FILE}，原始尺度未预标准化) ===")
    print(f"X={str(Xall.shape)} y={str(yall.shape)} p={pdim} 总样本={Xall.shape[0]} {'OK' if p_ok else 'FAIL'}")
    print("=" * 72)

    # ---- 单池切分（fixed seed 一次切好域；repeat 只变源/目标标签采样与目标测试）----
    pool_sources, pool_target = build_pool_split(args, args.seed)
    print(f"    [单池切分] 源域数={len(pool_sources)}, 每源域样本={args.n_source_samples}, "
          f"目标域剩余样本={len(pool_target[0])}, 噪声维数={getattr(args, 'noise_cols', 0)}")

    all_out = {}
    for tgt in TARGET_NAMES:
        res = {m: [] for m in ["lasso", "lasso_lab", "rand_nn", "fixed_target", "fixed_mean", "fixed_ow"]}
        res_empty = {m: [] for m in res}       # 选中的源域为空
        res_nonempty = {m: [] for m in res}    # 选中的源域非空
        sel_records = []
        for r in range(args.repeats):
            rep_seed = args.seed * (args.repeats + 1) + r
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)
            sources = pool_sources
            r2s, use_names = run_target_once(tgt, sources, args, rep_seed, base_target=pool_target)
            for m in res:
                res[m].append(r2s[m])
            if len(use_names) == 0:
                for m in r2s: res_empty[m].append(r2s[m])
            else:
                for m in r2s: res_nonempty[m].append(r2s[m])
            sel_records.append(use_names)
        all_out[tgt] = {"metrics": res, "selected_sources": sel_records}

        print(f"\n===== 目标域: {tgt} (seed={args.seed}) =====")
        for m, arr in res.items():
            a = np.array(arr, dtype=float)
            print(f"  {m:12s} R2 = {np.nanmean(a):.4f} ± {np.nanstd(a):.4f}")
        # ---- 按选源域是否为空 分组统计 ----
        n_empty = len(res_empty["lasso"]); n_ne = len(res_nonempty["lasso"])
        print(f"\n  [按选源域是否为空分组] 空源域={n_empty}/{args.repeats}, 非空源域={n_ne}/{args.repeats}")
        for tag, dd in [("空源域", res_empty), ("非空源域", res_nonempty)]:
            if len(res_nonempty["lasso"]) if tag == "非空源域" else len(res_empty["lasso"]):
                for m in res:
                    a = np.array(dd[m], dtype=float)
                    print(f"    {m:12s} ({tag}): R2={np.nanmean(a):.4f} ± {np.nanstd(a):.4f}")
        print("  源域被选中频率（重复间累计）:")
        for g in pool_sources.keys():
            c = sum(1 for s in sel_records if g in s)
            print(f"    {g:14s} {c:3d}/{args.repeats} = {c/args.repeats:.0%}")
    return all_out

# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                   help="全局随机种子（每次 repeat 派生确定性种子固定 np 与 torch）")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--n_source_domains", type=int, default=2,
                   help="单池切分：源域数目 K")
    p.add_argument("--n_source_samples", type=int, default=3950,
                   help="单池切分：每个源域的样本数；共 K*ns 个做源域，剩余全部归目标域")
    p.add_argument("--noise_cols", type=int, default=0,
                   help="追加的独立标准高斯噪声列数(制造高维稀疏冗余特征); 0=不加纯真实8维")
    p.add_argument("--n_target_samples", type=int, default=100,
                   help="目标域训练集样本数(再拆有标签/无标签); 0=用剩余全部(默认)")
    p.add_argument("--n_label_target_test", type=int, default=192,
                   help="目标域测试集独立抽取样本数")
    p.add_argument("--target_labeled_ratio", type=float, default=0.5,
                   help="目标有标签比例; 0.06*500≈30标签(极稀缺), 其余为大量无标签")
    p.add_argument("--source_labeled_ratio", type=float, default=0.5)
    p.add_argument("--kef_lam", type=float, default=1e-5)
    p.add_argument("--no_selection", action="store_false", dest="use_selection", default=True,
                   help="关闭源域选择，直接用全部源域转移（默认开启）")
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--fdr_control", action="store_true", default=True,
                   help="[默认] 选域时用 FDR_BH 校正")
    p.add_argument("--dist_norm", type=str, default="inf",
                   choices=["l1", "l2", "inf"],
                   help="选域时域间方向差异的范数(默认 l1; inf = 原版 max)")
    p.add_argument("--sparse_k", type=int, default=0,
                   help="选域前把 index 方向 top-k 稀疏化; 0=不稀疏")
    p.add_argument("--no_fdr", action="store_false", dest="fdr_control",
                   help="关闭 FDR 校正，直接用 alpha 判显著")
    p.add_argument("--nn_hidden", type=int, default=64)
    p.add_argument("--nn_epochs", type=int, default=2000)
    p.add_argument("--nn_lr", type=float, default=0.002)
    p.add_argument("--log_file", type=str, default=None,
                   help="日志文件路径(默认自动命名)")
    p.add_argument("--target_city", type=str, default="1",
                   help="目标域(1/2/3); 其余域自动作为源域")
    p.add_argument("--single_file", type=str, default="puma8NH_raw_arff.npz",
                   help="Communities 真实犯罪单文件(100维, 3个自造域 1/2/3)")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="关闭详细过程打印(仍保留汇总输出)")
    args = p.parse_args()

    # 配置文件后缀：区分源域数/源样本量/噪声维数/标签比例，使不同配置的结果与日志不互相覆盖
    cfg = (f"_K{args.n_source_domains}_ns{args.n_source_samples}"
           f"_nc{args.noise_cols}_tlr{args.target_labeled_ratio}")
    log_filename = args.log_file or f"real_data_log_seed{args.seed}_rep{args.repeats}{cfg}.log"

    class Tee:
        def __init__(self, filename):
            self.file = open(filename, "w", encoding="utf-8", buffering=1)
            self.stdout = sys.stdout
        def write(self, message):
            self.stdout.write(message); self.stdout.flush()
            self.file.write(message); self.file.flush()
        def flush(self):
            self.stdout.flush(); self.file.flush()
        def close(self):
            self.file.close()

    tee = Tee(log_filename)
    sys.stdout = tee
    try:
        print("=" * 72)
        print(f"单池迁移实验: 目标=剩余全部样本, 源域数={args.n_source_domains} × 每源域样本={args.n_source_samples}")
        print(f"补噪噪声维数 noise_cols={args.noise_cols} (真实特征之外追加的独立标准高斯列数)")
        print(f"数据=原始尺度(未预标准化); 各方法在各自域内做内部标准化; 域划分固定, repeat 仅重抽标签/测试")
        print(f"目标域: 训练 n={args.n_target_samples} (有标签×{args.target_labeled_ratio}), "
              f"测试 n={args.n_label_target_test}")
        print(f"全局种子 seed={args.seed}（每 repeat 派生确定性种子，np+torch 均固定）")
        print(f"选域距离范数 dist_norm={args.dist_norm}, sparse_k={args.sparse_k}")
        print("=" * 72)
        res = run_experiment(args)
        with open(f"real_data_r2_results{cfg}_seed{args.seed}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n已保存 real_data_r2_results{cfg}_seed{args.seed}.json (含每 REPEAT 的 R2 与选中源域)")
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f"日志已写入 {log_filename}")
