#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
源域选择实验：自适应源域选择（KEF + 置换检验，优化版）
- 方向估计：KEF（核特征估计），半监督框架
- 置换检验：得分矩阵仅计算一次，置换时复用，速度提升约 100 倍
- 更严谨版本：零分布同时置换源域和目标域标签（双重置换）
"""

import os, sys, copy, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import argparse, pickle as pkl
import datetime

# ---------- KEF 导入 ----------
try:
    from kscore.estimators import Tikhonov
    from kscore.kernels import CurlFreeIMQ
except ImportError:
    raise ImportError("请安装 numethod 包 (pip install numethod)")

# ---------- 数据生成 ----------
def ar1_cov(d, rho, min_eig=1e-6):
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov[i, j] = rho ** abs(i - j)
    eig = np.linalg.eigvals(cov)
    if np.min(eig) < min_eig:
        cov += (min_eig - np.min(eig) + 1e-6) * np.eye(d)
    return cov

def generate_data(a, n, dom, sde, nsf, ar_rho=0.6):
    cov = ar1_cov(len(a), ar_rho)
    sample = np.random.multivariate_normal(np.zeros_like(a), cov, size=n)
    inp = sample @ a
    out = nsf[dom](inp)
    error = np.random.normal(0, sde, n)
    out += error
    return out, sample

def gen_a0(d, s, l, u):
    sign = np.random.binomial(1, 0.5, s) * 2 - 1
    a0 = np.zeros(d)
    nz = np.random.uniform(l, u, s)
    nz = sign * nz
    a0[:s] = nz
    a0 = a0 / np.linalg.norm(a0)
    return a0

def gen_ak(a0, s, mode='normal', n_dif=1, rho_perturb=0.999,
           n_spike=2, fixed_shift=2.0, directional=True):
    if mode == 'normal':
        ak = copy.deepcopy(a0)
        id1 = np.random.randint(0, s, n_dif)
        id2 = np.random.randint(s, len(a0), n_dif)
        ak[id2] = ak[id1]
        ak[id1] = rho_perturb * ak[id1]
        ak = ak / np.linalg.norm(ak)
        return ak
    elif mode == 'directional_fixed':
        ak = copy.deepcopy(a0)
        all_indices = np.arange(len(a0))
        spike_idx = np.random.choice(all_indices, n_spike, replace=False)
        for idx in spike_idx:
            sign = np.sign(ak[idx]) if directional else np.random.choice([-1, 1])
            if sign == 0:
                sign = 1
            ak[idx] += sign * fixed_shift
        ak = ak / np.linalg.norm(ak)
        return ak
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ---------- KEF 相关函数 ----------
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

# ---------- 半监督方向估计 ----------
def estimate_direction_from_score(score, X_label, y_label, grids_beta, dom):
    """
    给定预先计算好的得分矩阵 score，估计方向。
    dom=0：目标域（软阈值），dom=1：源域（硬阈值）
    """
    score = np.clip(score, np.quantile(score, 0.1), np.quantile(score, 0.9))
    if dom == 0:
        lam_sel = sel_tun_lam(grids_beta, min(5, len(y_label)//2), y_label.reshape(-1,1), score, 0)
        raw = np.mean(y_label.reshape(-1,1) * score, axis=0)
        dir_vec = st(raw, lam_sel)
    else:  # dom=1
        lam_sel = sel_tun_lam(grids_beta, 5, y_label.reshape(-1,1), score, 1)
        raw = np.mean(y_label.reshape(-1,1) * score, axis=0)
        dir_vec = ht(raw, lam_sel)
    
    if np.linalg.norm(dir_vec) == 0:
        dir_vec = np.mean(y_label.reshape(-1,1) * score, axis=0)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    return dir_vec

def estimate_base_direction_kef_semi(X_unlabel, X_label, y_label, lam=1e-4, grids_beta=None):
    if grids_beta is None:
        grids_beta = np.logspace(-4, 1, 20)
    score = estimate_kef_gradients(X_unlabel, X_label, lam=lam)
    return estimate_direction_from_score(score, X_label, y_label, grids_beta, dom=0)

# ===================================================================
# 核心改动：更严谨的置换检验（两者都置换）
# ===================================================================
def adaptive_selection_by_permutation_double_fast(
    X_sources_unlabel, X_sources_label, y_sources_label,
    X_target_unlabel, X_target_label, y_target_label,
    target_dir,  # 注意：这个参数仍然需要传入用于符号对齐，但不作为固定靶子
    lam=1e-4, grids_beta=None,
    n_perm=200, alpha=0.05, fdr_control=True
):
    """
    更严谨的双重置换检验：
    零分布通过同时置换源域和目标域的标签生成。
    这样零分布包含了双方估计误差，p值更准确。
    """
    if grids_beta is None:
        grids_beta = np.logspace(-4, 1, 20)
    
    K = len(X_sources_label)
    p_values = np.ones(K)
    obs_dist = np.zeros(K)

    # ---- 预计算：所有源域的得分矩阵（复用） ----
    source_scores = []
    for i in range(K):
        score = estimate_kef_gradients(
            X_sources_unlabel[i], 
            X_sources_label[i], 
            lam=lam
        )
        source_scores.append(score)

    # ---- 预计算：目标域的得分矩阵（复用） ----
    target_score = estimate_kef_gradients(
        X_target_unlabel, 
        X_target_label, 
        lam=lam
    )

    # ---- 对每个源域进行检验 ----
    for i in range(K):
        X_label_src = X_sources_label[i]
        y_label_src = y_sources_label[i]
        score_src = source_scores[i]
        n_src = len(y_label_src)
        n_tgt = len(y_target_label)

        # 观测方向（真实标签）
        a_obs_src = estimate_direction_from_score(score_src, X_label_src, y_label_src, grids_beta, dom=1)
        # 符号对齐到目标方向（仅用于对齐，不参与零分布）
        if np.dot(a_obs_src, target_dir) < 0:
            a_obs_src = -a_obs_src
        obs_dist[i] = np.max(np.abs(a_obs_src - target_dir))

        # ---- 双重置换生成零分布 ----
        null_dists = []
        for _ in range(n_perm):
            # 1. 置换源域标签
            y_perm_src = y_label_src[np.random.permutation(n_src)]
            a_perm_src = estimate_direction_from_score(score_src, X_label_src, y_perm_src, grids_beta, dom=1)

            # 2. 置换目标域标签（关键：目标方向也随机化）
            y_perm_tgt = y_target_label[np.random.permutation(n_tgt)]
            a_perm_tgt = estimate_direction_from_score(target_score, X_target_label, y_perm_tgt, grids_beta, dom=0)

            # 3. 符号对齐到 target_dir（统一参考系）
            if np.dot(a_perm_src, target_dir) < 0:
                a_perm_src = -a_perm_src
            if np.dot(a_perm_tgt, target_dir) < 0:
                a_perm_tgt = -a_perm_tgt

            # 4. 计算两个纯随机方向的距离
            d_perm = np.max(np.abs(a_perm_src - a_perm_tgt))
            null_dists.append(d_perm)

        p_values[i] = np.mean(np.array(null_dists) <= obs_dist[i])

    # 多重比较校正
    if fdr_control:
        from statsmodels.stats.multitest import multipletests
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        selected = np.where(reject)[0].tolist()
    else:
        selected = np.where(p_values < alpha)[0].tolist()

    return selected, p_values, obs_dist


# ---------- 主实验函数 ----------
def run_experiment(args):
    d = args.dim
    s = args.sparsity
    n_good = args.n_good
    n_bad = args.n_bad
    K = n_good + n_bad
    sde = args.noise_std
    l, u = 0.9, 1
    ar_rho = args.rho
    rep = args.repeats
    n_target_total = args.n_target_samples
    n_source_total = args.n_source_total
    kef_lam = args.kef_lam
    n_perm = args.n_perm
    alpha = args.alpha
    fdr_control = args.fdr_control
    target_labeled_ratio = args.target_labeled_ratio
    source_labeled_ratio = args.source_labeled_ratio

    # 非线性函数
    if args.shift_mode == 'same':
        nsf = [lambda x: np.tanh(x) for _ in range(K+1)]
        print("使用相同连接函数：所有域 tanh(x)")
    elif args.shift_mode == 'contrast':
        contract = args.large_shift
        nsf = [lambda x: np.tanh(x)]  # 目标域无偏移
        if n_good > 0:
            good_offsets = np.linspace(contract / n_good, contract, n_good)
        else:
            good_offsets = np.array([])
        if n_bad > 0:
            bad_offsets = np.linspace(contract / n_bad, contract, n_bad)
        else:
            bad_offsets = np.array([])
        offsets = np.concatenate([good_offsets, bad_offsets])
        for off in offsets:
            nsf.append(lambda x, s=off: np.tanh(x - s))
        print(f"对比模式：good 偏移 = {good_offsets}, bad 偏移 = {bad_offsets}")
        source_types = ['good'] * n_good + ['bad'] * n_bad
    else:
        raise ValueError(f"Unknown shift_mode: {args.shift_mode}")

    grids_beta = np.logspace(-4, 1, 20)

    param_tag = (
        f"{args.n_good}g_{args.n_bad}b"
        f"_nt{args.n_target_samples}_ns{args.n_source_total}"
        f"_fs{args.fixed_shift}_ls{args.large_shift}_{args.shift_mode}"
    )

    interim_file = f"interim_kef_semi_{param_tag}_{args.repeats}rep.pkl"


    if os.path.exists(interim_file):
        try:
            with open(interim_file, 'rb') as f:
                interim_data = pkl.load(f)
            algo_adaptive_correct = interim_data['algo_adaptive_correct']
            per_repeat_results = interim_data['per_repeat_results']
            start_rep = len(per_repeat_results)
            print(f"从中间文件恢复，已完成 {start_rep} 次重复")
        except Exception as e:
            print(f"加载中间文件失败: {e}，从头开始")
            algo_adaptive_correct = 0
            per_repeat_results = []
            start_rep = 0
    else:
        algo_adaptive_correct = 0
        per_repeat_results = []
        start_rep = 0

    # 逐源域统计
    per_domain_TP = [0] * K
    per_domain_FN = [0] * K
    per_domain_FP = [0] * K
    per_domain_TN = [0] * K

    if start_rep >= rep:
        print(f"已完成全部 {rep} 次重复，跳过实验")
    else:
        print(f"从第 {start_rep+1} 次重复开始继续实验")

    for r in range(start_rep, rep):
        print(f"\n========== Replication {r+1}/{rep} ==========")
        np.random.seed(r)
        torch.manual_seed(r)

        # ---------- 生成真实参数 ----------
        a0 = gen_a0(d, s, l, u)
        source_dirs_true = []
        source_types = []
        for _ in range(n_good):
            ak = gen_ak(a0, s, mode='normal', n_dif=1, rho_perturb=0.995)
            source_dirs_true.append(ak)
            source_types.append('good')
        for _ in range(n_bad):
            ak = gen_ak(a0, s, mode='directional_fixed', n_spike=args.n_spike,
                        fixed_shift=args.fixed_shift, directional=True)
            source_dirs_true.append(ak)
            source_types.append('bad')

        print("真实源域方向与目标域方向夹角:")
        for i, true_dir in enumerate(source_dirs_true):
            angle = np.arccos(np.clip(np.dot(true_dir, a0), -1, 1)) * 180 / np.pi
            print(f"  源域 {i+1} ({source_types[i]}): Angle={angle:.2f}°")

        # ---------- 生成目标域数据并划分 ----------
        y_all_target, X_all_target = generate_data(a0, n_target_total, 0, sde, nsf, ar_rho)
        n_target_total_actual = len(X_all_target)
        perm_target = np.random.permutation(n_target_total_actual)
        n_labeled_target = int(n_target_total_actual * target_labeled_ratio)
        label_idx_target = perm_target[:n_labeled_target]
        unlabel_idx_target = perm_target[n_labeled_target:]
        X_label_target = X_all_target[label_idx_target]
        y_label_target = y_all_target[label_idx_target]
        X_unlabel_target = X_all_target[unlabel_idx_target]

        print(f"目标域: 总 {n_target_total_actual}, 有标签 {len(X_label_target)}, 无标签 {len(X_unlabel_target)}")

        # ---------- 生成源域数据并划分 ----------
        source_label_data = []
        source_unlabel_data = []
        for kk in range(K):
            y_all_src, X_all_src = generate_data(source_dirs_true[kk], n_source_total, kk+1, sde, nsf, ar_rho)
            n_src_total = len(X_all_src)
            perm_src = np.random.permutation(n_src_total)
            n_labeled_src = int(n_src_total * source_labeled_ratio)
            label_idx_src = perm_src[:n_labeled_src]
            unlabel_idx_src = perm_src[n_labeled_src:]
            X_label_src = X_all_src[label_idx_src]
            y_label_src = y_all_src[label_idx_src]
            X_unlabel_src = X_all_src[unlabel_idx_src]
            source_label_data.append((X_label_src, y_label_src))
            source_unlabel_data.append(X_unlabel_src)
            print(f"源域 {kk+1}: 总 {n_src_total}, 有标签 {len(X_label_src)}, 无标签 {len(X_unlabel_src)}")

        # ========== 1. 估计目标方向 ==========
        print("估计目标方向（半监督 KEF）...")
        target_dir = estimate_base_direction_kef_semi(X_unlabel_target, X_label_target, y_label_target,
                                                      lam=kef_lam, grids_beta=grids_beta)
        print(f"目标方向估计完成，L2 范数: {np.linalg.norm(target_dir):.4f}")

        # ========== 2. 自适应选择（双重置换检验） ==========
        print("\n执行自适应源域选择（双重置换检验）...")
        X_sources_unlabel = source_unlabel_data
        X_sources_label = [X for X, _ in source_label_data]
        y_sources_label = [y for _, y in source_label_data]
        selected_adaptive, p_values, obs_dist = adaptive_selection_by_permutation_double_fast(
            X_sources_unlabel, X_sources_label, y_sources_label,
            X_unlabel_target, X_label_target, y_label_target,
            target_dir,  # 仅用于符号对齐，不作为固定靶子
            lam=kef_lam, grids_beta=grids_beta,
            n_perm=n_perm, alpha=alpha, fdr_control=fdr_control
        )
        good_indices = [i for i, t in enumerate(source_types) if t == 'good']
        adaptive_exact = set(selected_adaptive) == set(good_indices)
        if adaptive_exact:
            algo_adaptive_correct += 1

        # 逐源域更新混淆矩阵
        selected_set = set(selected_adaptive)
        for i in range(K):
            if source_types[i] == 'good':
                if i in selected_set:
                    per_domain_TP[i] += 1
                else:
                    per_domain_FN[i] += 1
            else:  # 'bad'
                if i in selected_set:
                    per_domain_FP[i] += 1
                else:
                    per_domain_TN[i] += 1

        # 打印本轮每个域的表现
        print("\n--- 本轮逐源域判断情况 ---")
        for i in range(K):
            if source_types[i] == 'good':
                status = "被选中" if i in selected_set else "被漏掉"
            else:
                status = "被误选" if i in selected_set else "正确拒绝"
            print(f"  源域 {i+1} ({source_types[i]}): {status}")

        print("\n--- 选择结果 ---")
        print(f"自适应选择 (p<{alpha}, FDR={fdr_control}): {selected_adaptive}, 精确匹配={adaptive_exact}")
        print(f"p 值: {p_values}")
        print(f"观测距离: {obs_dist}")

        # 记录
        per_repeat_results.append({
            'rep': r+1,
            'adaptive_exact': adaptive_exact,
            'selected': selected_adaptive
        })

        # 保存中间结果
        interim_data = {
            'algo_adaptive_correct': algo_adaptive_correct,
            'per_repeat_results': per_repeat_results
        }
        with open(interim_file, 'wb') as f:
            pkl.dump(interim_data, f)
        print(f"中间结果已保存至 {interim_file} (已运行 {r+1} 次重复)")

    # ---- 最终汇总 ----
    print("\n========== 整轮实验精确匹配正确率 ==========")
    print(f"自适应选择 (双重置换检验, alpha={alpha}, FDR={fdr_control}): {algo_adaptive_correct}/{rep} = {algo_adaptive_correct/rep:.2%}")

    print("\n========== 每个源域独立判断概率（跨重复汇总） ==========")
    for i in range(K):
        if source_types[i] == 'good':
            total_cases = per_domain_TP[i] + per_domain_FN[i]
            if total_cases > 0:
                prob = per_domain_TP[i] / total_cases
                print(f"  源域 {i+1} (真实类型: 好) -> 被判断为'好'的概率: {prob:.2%}  ({per_domain_TP[i]}/{total_cases})")
            else:
                print(f"  源域 {i+1} (真实类型: 好) -> 无统计样本")
        else:
            total_cases = per_domain_TN[i] + per_domain_FP[i]
            if total_cases > 0:
                prob = per_domain_TN[i] / total_cases
                print(f"  源域 {i+1} (真实类型: 坏) -> 被判断为'坏'的概率: {prob:.2%}  ({per_domain_TN[i]}/{total_cases})")
            else:
                print(f"  源域 {i+1} (真实类型: 坏) -> 无统计样本")

    output_data = {
        'algo_adaptive_correct': algo_adaptive_correct,
        'total_repeats': rep,
        'per_repeat': per_repeat_results,
        'alpha': alpha,
        'fdr_control': fdr_control,
        'target_labeled_ratio': target_labeled_ratio,
        'source_labeled_ratio': source_labeled_ratio,
        'source_types': source_types,
        'per_domain_TP': per_domain_TP,
        'per_domain_FN': per_domain_FN,
        'per_domain_FP': per_domain_FP,
        'per_domain_TN': per_domain_TN
    }
    final_file = f"exact_match_kef_semi_{param_tag}_rep{rep}.pkl"
    with open(final_file, 'wb') as f:
        pkl.dump(output_data, f)
    print(f"\n最终结果已保存至 {final_file}")

    if os.path.exists(interim_file):
        os.remove(interim_file)
        print(f"中间文件 {interim_file} 已删除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--sparsity", type=int, default=30)
    parser.add_argument("--n_good", type=int, default=2)
    parser.add_argument("--n_bad", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=1/4)
    parser.add_argument("--rho", type=float, default=1/8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n_target_samples", type=int, default=500, help="目标域总样本数")
    parser.add_argument("--n_source_total", type=int, default=6000,
                        help="每个源域的总样本数（包含有标签和无标签，按 source_labeled_ratio 划分）")
    parser.add_argument("--n_spike", type=int, default=1)
    parser.add_argument("--fixed_shift", type=float, default=0.6)
    parser.add_argument("--shift_mode", type=str, default="contrast", choices=["same", "contrast"])
    parser.add_argument("--large_shift", type=float, default=0.5)
    parser.add_argument("--log_file", type=str, default=None)
    # KEF 参数
    parser.add_argument("--kef_lam", type=float, default=1e-5, help="KEF 正则化参数")
    # 置换检验参数
    parser.add_argument("--n_perm", type=int, default=200, help="每个源域的置换次数")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    parser.add_argument("--no_fdr_control", action="store_false", dest="fdr_control", default=True,
                    help="禁用 FDR 校正（Benjamini-Hochberg）")
    # 半监督比例参数
    parser.add_argument("--target_labeled_ratio", type=float, default=0.5,
                        help="目标域有标签比例")
    parser.add_argument("--source_labeled_ratio", type=float, default=0.5,
                        help="源域有标签比例")
    args = parser.parse_args()

    param_tag = (
        f"{args.n_good}g_{args.n_bad}b"
        f"_nt{args.n_target_samples}_ns{args.n_source_total}"
        f"_fs{args.fixed_shift}_ls{args.large_shift}_{args.shift_mode}"
    )

    if args.log_file is None:
        log_filename = f"exact_match_kef_semi_opt_{param_tag}_rep{args.repeats}.log"
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
    print(f"开始时间: {datetime.datetime.now()}")
    print("="*80)
    try:
        run_experiment(args)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"结束时间: {datetime.datetime.now()}")
        sys.stdout = tee.stdout
        tee.close()
