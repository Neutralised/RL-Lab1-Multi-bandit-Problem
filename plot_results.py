"""plot_results.py —— 读取 results/*.csv 并绘制实验图表。

用法:
    python plot_results.py            # 读 results/，图输出到 docs/assets/figures/

依赖: numpy, matplotlib（不需要 pandas）
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 自带雅黑；没有则回退 DejaVu Sans，中文会显示为方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "docs", "assets", "figures")

# 算法 → 颜色/线型，保证同一算法在所有图里保持一致
ALGO_COLOR = {
    "greedy": "C3",
    "eps-greedy 0.01": "C0",
    "eps-greedy 0.05": "C2",
    "eps-greedy 0.1": "C1",
    "reducing-eps-greedy 0.05": "C4",
    "UCB 0.5": "C5",
    "UCB sqrt2": "C6",
    "UCB 2": "C7",
}
ALGO_STYLE = {
    "greedy": "--",
    "reducing-eps-greedy 0.05": "-.",
    "UCB 2": ":",
}


# ------------------------------------------------------------------ 读 CSV
def read_curves(path):
    """{(exp, env, algo, param_name, T): {'t','mean','se','opt'}}"""
    out = defaultdict(lambda: {"t": [], "mean": [], "se": [], "opt": []})
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["experiment"], row["env"], row["algo"], row["param_name"], int(row["T"]))
            out[key]["t"].append(int(row["t"]))
            out[key]["mean"].append(float(row["mean_regret"]))
            out[key]["se"].append(float(row["se_regret"]))
            out[key]["opt"].append(float(row["opt_ratio"]))
    for v in out.values():
        for k in ("t", "mean", "se", "opt"):
            v[k] = np.asarray(v[k], dtype=float)
    return out


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_nsub(path):
    """E3 的次优臂平均选择次数：{(algo, gap): count}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == "E3_gap" and int(row["arm"]) == 1:
                out[(row["algo"], float(row["param_value"]))] = float(row["count"])
    return out


# ------------------------------------------------------------------ 小工具
def select(summary, exp, env=None, algo=None):
    rows = [r for r in summary if r["experiment"] == exp]
    if env is not None:
        rows = [r for r in rows if r["env"] == env]
    if algo is not None:
        rows = [r for r in rows if r["algo"] == algo]
    return rows


def final_regret(summary, exp, env, algo, param_value=None, T=None):
    for r in select(summary, exp, env, algo):
        if param_value is not None and abs(float(r["param_value"]) - float(param_value)) > 1e-9:
            continue
        if T is not None and int(r["T"]) != T:
            continue
        return float(r["final_regret"])
    raise KeyError((exp, env, algo, param_value, T))


def band(ax, curve, label, color, ls="-", lw=1.4):
    t, m, se = curve["t"], curve["mean"], curve["se"]
    ax.plot(t, m, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0)


# ------------------------------------------------------------------ E1 / E6
def fig_algorithms(curves, env, fname, title, T=2000):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.subplots_adjust(wspace=0.28)
    for algo, color in ALGO_COLOR.items():
        key = ("E1_algorithms" if env == "bernoulli" else "E6_gaussian", env, algo, "-", T)
        if key not in curves:
            continue
        ls = ALGO_STYLE.get(algo, "-")
        band(axes[0], curves[key], algo, color, ls)
        axes[1].plot(curves[key]["t"], curves[key]["opt"], color=color, ls=ls, lw=1.3, label=algo)
    axes[0].set(xlabel="轮次 $t$", ylabel=r"累积 pseudo-regret $\bar R_T(t)$",
                title=f"{title}：regret 曲线（均值 ± 95% 置信带）")
    axes[1].set(xlabel="轮次 $t$", ylabel="累计最优臂占比", title=f"{title}：最优臂占比", ylim=(0, 1.02))
    axes[1].yaxis.labelpad = 10
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)


# ------------------------------------------------------------------ E2
def fig_horizon(curves, summary, algs, Ts):
    fig, axes = plt.subplots(1, len(algs), figsize=(4.8 * len(algs), 4.4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(Ts)))
    for ax, algo in zip(axes, algs):
        # 先画长 T、后画短 T，避免短曲线被长曲线的置信带遮住
        for c, T in zip(colors[::-1], sorted(Ts, reverse=True)):
            key = ("E2_horizon", "bernoulli", algo, "T", T)
            if key in curves:
                # 不同 T 是同一条配置的前缀，置信带会互相遮挡，这里只画均值曲线
                ax.plot(curves[key]["t"], curves[key]["mean"], color=c, lw=1.5, label=f"$T={T}$")
        ax.set(xlabel="轮次 $t$", ylabel=r"$\bar R_T(t)$", title=algo)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(r"E2：只改时间步 $T$（$K=10$ 伯努利，runs=200）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "e2_horizon.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for algo in algs:
        rows = sorted(select(summary, "E2_horizon", "bernoulli", algo), key=lambda r: int(r["T"]))
        xs = [int(r["T"]) for r in rows]
        axes[0].plot(xs, [float(r["final_regret"]) for r in rows], "o-",
                     color=ALGO_COLOR[algo], label=algo)
        axes[1].plot(xs, [float(r["regret_per_t"]) for r in rows], "o-",
                     color=ALGO_COLOR[algo], label=algo)
    axes[0].set(xlabel="$T$", ylabel=r"$\bar R_T$", title=r"终值遗憾 vs $T$")
    axes[1].set(xlabel="$T$", ylabel=r"$\bar R_T/T$", title=r"平均每轮损失 vs $T$")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUTDIR, "e2_horizon_summary.png"))
    plt.close(fig)


# ------------------------------------------------------------------ E3
def fig_gap(summary, nsub, algs, gaps, T=5000):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for algo in algs:
        R = [final_regret(summary, "E3_gap", "bernoulli", algo, g, T=T) for g in gaps]
        C = [nsub[(algo, g)] for g in gaps]
        axes[0].plot(gaps, R, "o-", color=ALGO_COLOR[algo], label=algo)
        axes[1].plot(gaps, C, "s-", color=ALGO_COLOR[algo], label=algo)
    axes[0].set(xlabel=r"均值差距 $\Delta$", ylabel=r"$\bar R_T$（$T=5000$）",
                title=r"E3：$\Delta$ 变小时总遗憾反而变小")
    axes[1].set(xlabel=r"均值差距 $\Delta$", ylabel=r"$N_{\mathrm{sub}}(T)$",
                title=r"E3：次优臂被选次数（$T/2=2500$ ≈ 完全分不清）")
    axes[1].axhline(T / 2, color="gray", ls=":", lw=1, label="每轮随机选择的期望")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUTDIR, "e3_gap.png"))
    plt.close(fig)


# ------------------------------------------------------------------ E4 / E5
def fig_sensitivity(summary, exp, xlabel, fname, title, ref=None):
    rows = select(summary, exp, "bernoulli")
    rows = sorted(rows, key=lambda r: float(r["param_value"]))
    xs = [float(r["param_value"]) for r in rows]
    ys = [float(r["final_regret"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(range(len(xs)), ys, "o-", color="C0", lw=1.6)
    best = int(np.argmin(ys))
    ax.plot([best], [ys[best]], "o", ms=11, mfc="none", mec="C3", mew=2,
            label=f"最优 {xlabel}={xs[best]:g}，$\\bar R_T$={ys[best]:.1f}")
    if ref is not None:
        ax.axhline(ref, color="C3", ls="--", lw=1, label=f"greedy 参考（{ref:.1f}）")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
    ax.set(xlabel=xlabel, ylabel=r"$\bar R_T$（$T=2000$）", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    curves = read_curves(os.path.join(RESULTS, "curves.csv"))
    summary = read_summary(os.path.join(RESULTS, "summary.csv"))
    nsub = read_nsub(os.path.join(RESULTS, "counts.csv"))

    algos_few = ["greedy", "eps-greedy 0.05", "UCB sqrt2"]
    gaps = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    fig_algorithms(curves, "bernoulli", "e1_algorithms_bernoulli.png", "E1 伯努利环境")
    fig_algorithms(curves, "gaussian", "e6_gaussian.png", "E6 高斯环境")
    fig_horizon(curves, summary, algos_few, [200, 500, 1000, 2000, 5000, 10000])
    fig_gap(summary, nsub, algos_few, gaps)

    greedy_2000 = final_regret(summary, "E4_epsilon", "bernoulli", "eps-greedy", 0.0, T=2000)
    fig_sensitivity(summary, "E4_epsilon", r"$\varepsilon$", "e4_epsilon.png",
                    r"E4：$\varepsilon$ 的影响（$K=10$，$T=2000$）", ref=greedy_2000)
    fig_sensitivity(summary, "E5_ucb_c", r"$c$", "e5_ucb_c.png",
                    r"E5：UCB 的 $c$ 的影响（$K=10$，$T=2000$）")

    print("图已输出到", OUTDIR)
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith(".png"):
            print("  ", f)


if __name__ == "__main__":
    main()
