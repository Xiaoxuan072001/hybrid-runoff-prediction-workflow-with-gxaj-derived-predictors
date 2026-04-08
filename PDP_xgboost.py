import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib

# === 字体设置：优先 Times New Roman，mathtext 用直立罗马体风格（stix） ===
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 18
# mathtext settings to render upright roman (compatible with Times-like fonts)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# === 读取数据 ===
df = pd.read_csv("LSTM_normalized.csv")
exclude_vars = ["Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg",
                "ISA", "LU(snow)", "K", "CI"]

X_all = df.drop(columns=["Q_obs", "ID", "date"])
X_all = X_all.drop(columns=[c for c in exclude_vars if c in X_all.columns])
y_all = df["Q_obs"]

# === 数据划分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# === 模型训练 ===
xgb_model = XGBRegressor(
    max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7,
    reg_alpha=5.0, reg_lambda=10.0, n_estimators=300,
    objective="reg:squarederror", verbosity=0
)
xgb_model.fit(X_train, y_train)

# === 要绘制的特征 ===
features_to_plot = ["Q_XAJ", "NDVI", "Pt-1", "SM2", "LST", "AI", "Sproxy", "DEM_range", "Area"]

# === 反归一化所需的最大值、最小值与单位 ===
max_vals = {
    "Area": 10910.0, "SM2": 0.46889, "NDVI": 0.8194, "Pt-1": 221.15,
    "LST": 39.76, "DEM_range": 3690.0, "Q_XAJ": 2732.0, "AI": 75.18, "Sproxy": 0.746
}
min_vals = {
    "Area": 313.0, "SM2": 0.153, "NDVI": 0.01, "Pt-1":0,
    "LST": -4.06, "DEM_range": 343.0, "Q_XAJ": 0, "AI": 0, "Sproxy": 0
}
units = {
    "Area": "km²", "SM2": "m³/m³", "NDVI": "", "Pt-1": "mm", "LST": "°C",
    "DEM_range": "m", "Q_XAJ": "m³/s", "AI": "", "Sproxy": ""
}

# ==== pretty_name: 将特征名转换为 mathtext（直立罗马体 + 下标） ====
def pretty_name(feat):
    mapping = {
        "SM1": r"$\mathrm{SM}_{1}$",
        "SM2": r"$\mathrm{SM}_{2}$",
        "SM3": r"$\mathrm{SM}_{3}$",
        "SM4": r"$\mathrm{SM}_{4}$",
        "t": r"$\mathrm{t}$",
        "tmin": r"$\mathrm{t}_{\mathrm{min}}$",
        "tmax": r"$\mathrm{t}_{\mathrm{max}}$",
        "tmean": r"$\mathrm{t}_{\mathrm{mean}}$",
        "Pt": r"$\mathrm{P}_{\mathrm{t}}$",
        "Pt-1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "Pt_1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "P": r"$\mathrm{P}$",
        "Rn": r"$\mathrm{R}_{\mathrm{n}}$",
        "Td": r"$\mathrm{T}_{\mathrm{d}}$",
        "DEM_ave": r"$\mathrm{DEM}_{\mathrm{ave}}$",
        "DEM_range": r"$\mathrm{DEM}_{\mathrm{range}}$",
        "Slope_ave": r"$\mathrm{Slope}_{\mathrm{ave}}$",
        "TWI_ave": r"$\mathrm{TWI}_{\mathrm{ave}}$",
        "Sproxy": r"$\mathrm{S}_{\mathrm{proxy}}$",
        "Q_XAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QGXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "NDVI": r"$\mathrm{NDVI}$",
        "LST": r"$\mathrm{LST}$",
        "Area": r"$\mathrm{Area}$",
        "ET": r"$\mathrm{ET}$",
        "RH": r"$\mathrm{RH}$",
        "Wind": r"$\mathrm{Wind}$",
        "Solar": r"$\mathrm{Solar}$",
        "Thermal": r"$\mathrm{Thermal}$",
        "SoilC": r"$\mathrm{SoilC}$",
        "SoilD": r"$\mathrm{SoilD}$",
        "LLM": r"$\mathrm{LLM}$",
        "LUM": r"$\mathrm{LUM}$",
        "CG": r"$\mathrm{CG}$",
        "AI": r"$\mathrm{AI}$",
    }
    if feat in mapping:
        return mapping[feat]
    if "_" in feat:
        left, right = feat.split("_", 1)
        return rf"$\mathrm{{{left}}}_{{\mathrm{{{right}}}}}$"
    return rf"$\mathrm{{{feat}}}$"

# ==== 工具函数 ====
def denorm(xs_norm, feat):
    """反归一化：x = x_norm * (max - min) + min（使用min_vals而非默认0）"""
    return xs_norm * (max_vals[feat] - min_vals[feat]) + min_vals[feat]

def fmt_tick_value(v, feat):
    if not np.isfinite(v):
        return ""
    if feat in {"Q_XAJ", "Pt-1", "DEM_range", "Area"}:
        return f"{int(round(v))}"
    if feat == "LST":
        return f"{v:.1f}"
    return f"{v:.2f}"

def apply_clean_xticks(ax, feat, xmin, xmax):
    """为子图设置整洁的 x 轴刻度与标签（自动/合法 steps）"""
    if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
        xmin, xmax = 0.0, 1.0
    ax.set_xlim(xmin, xmax)

    if feat in {"Q_XAJ", "Pt-1", "DEM_range", "Area", "AI", "LST"}:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v))}" if np.isfinite(v) else ""))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        nd = 2
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.{nd}f}" if np.isfinite(v) else ""))

def compute_pdp_ice(model, X, feature, grid=30, percentiles=(0.05, 0.95),
                    ice_samples=400, seed=42):
    """
    在归一化域上计算：
    xs_norm: 网格 (按分位范围)
    pdp_vals: PDP 均值
    ice_mat:  ICE 矩阵 [n_samples, grid]
    """
    rng = np.random.RandomState(seed)
    x = X[feature].to_numpy()
    mask = np.isfinite(x)
    x_valid = x[mask]

    if x_valid.size >= 5:
        lo = np.nanquantile(x_valid, percentiles[0])
        hi = np.nanquantile(x_valid, percentiles[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = 0.0, 1.0
        if hi - lo < 1e-6:
            lo -= 1e-3; hi += 1e-3
        xs_norm = np.linspace(lo, hi, grid)
    else:
        xs_norm = np.linspace(0.0, 1.0, grid)

    idx_pool = np.where(mask)[0]
    if idx_pool.size == 0:
        return xs_norm, np.full_like(xs_norm, np.nan, dtype=float), np.empty((0, xs_norm.size))

    pick = min(ice_samples, idx_pool.size)
    idx = rng.choice(idx_pool, size=pick, replace=False)
    X_sub = X.iloc[idx].copy()

    ice = np.zeros((len(X_sub), xs_norm.size), dtype=float)
    for j, xv in enumerate(xs_norm):
        X_tmp = X_sub.copy()
        X_tmp[feature] = xv
        ice[:, j] = model.predict(X_tmp)

    pdp = np.nanmean(ice, axis=0)
    return xs_norm, pdp, ice

# ==== 绘图：3×3 面板；每格：左轴 PDP，右轴 ICE ====
fig, axs = plt.subplots(3, 3, figsize=(12, 10))
axs = axs.ravel()

for i, feat in enumerate(features_to_plot):
    xs_norm, pdp_vals, ice_vals = compute_pdp_ice(
        xgb_model, X_all, feat, grid=30, percentiles=(0.05, 0.95),
        ice_samples=600, seed=42
    )

    # 反归一化到真实物理量（使用修改后的denorm函数，包含min_vals）
    xs_real = denorm(xs_norm, feat) if feat in max_vals else xs_norm
    finite = np.isfinite(xs_real)
    if finite.sum() < 2:
        xs_real = np.linspace(0, 1, len(xs_norm))
        finite = np.isfinite(xs_real)
    xs_plot = xs_real[finite]

    # 对 pdp 做截取（以保证与 xs_plot 对齐）
    p_len = len(xs_plot)
    pdp_plot = pdp_vals[:p_len]

    # 轴
    ax1 = axs[i]          # 左：PDP
    ax2 = ax1.twinx()     # 右：ICE

    # PDP（左） - 直接绘制主线（不绘制置信区间）
    ax1.plot(xs_plot, pdp_plot, color="#6cc6d8", lw=2.6, label="PDP (mean)", zorder=3)
    ax1.set_ylabel("PDP", color="#6cc6d8", fontsize=18, fontweight="bold")

    # ICE（右）
    if ice_vals.size > 0:
        n_lines = min(35, ice_vals.shape[0])
        step = max(1, ice_vals.shape[0] // n_lines)
        for k_idx in range(0, ice_vals.shape[0], step):
            ax2.plot(xs_plot, ice_vals[k_idx, :p_len], color="#ee7564", alpha=0.55, lw=1.0, zorder=2)
    ax2.set_ylabel("ICE", color="#ee7564", fontsize=18, fontweight="bold")

    # 横轴刻度（真实值 + 单位）
    xmin, xmax = float(np.nanmin(xs_plot)), float(np.nanmax(xs_plot))
    apply_clean_xticks(ax1, feat, xmin, xmax)
    apply_clean_xticks(ax2, feat, xmin, xmax)  # 保持两轴一致

    unit = units.get(feat, "")
    # 使用 pretty_name 渲染特征名（mathtext），并在后面显示单位（普通文本）
    if unit:
        ax1.set_xlabel(f"{pretty_name(feat)} ({unit})", fontsize=16, fontweight="bold")
    else:
        ax1.set_xlabel(f"{pretty_name(feat)}", fontsize=16, fontweight="bold")

    # 网格 & 标题（标题使用 pretty_name）
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_title(pretty_name(feat), fontsize=18, fontweight="bold")

    # y 轴范围按分位数自适应，防止“贴底”
    def _set_y(ax):
        lines = ax.get_lines()
        if not lines:
            return
        ys = []
        for ln in lines:
            y = ln.get_ydata()
            ys.append(y[np.isfinite(y)])
        if ys:
            y_all = np.concatenate(ys)
            if y_all.size:
                lo, hi = np.nanpercentile(y_all, [5, 95])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    pad = 0.06 * (hi - lo)
                    ax.set_ylim(lo - pad, hi + pad)
    _set_y(ax1); _set_y(ax2)

    # 小图例（避免覆盖数据）：只在第一列最后一格放图例或在每格下方注释
    if i == 0:
        ax1.legend(loc='upper left', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
outfn = "figure9PDP_ICE_dual_axis_times_noCI.png"
plt.savefig(outfn, dpi=600)
plt.close()
print(f"[完成] PDP+ICE 双纵轴图（无置信区间）已保存为：{outfn}")
