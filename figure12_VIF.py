import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------- config ----------
DATA_CSV = "LSTM_normalized.csv"
OUT_DIR = "figure12corr_vif_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

exclude_vars = ["Kes", "C", "CS", "Kech", "Kei", "Keg",
                "Xech", "Xes", "Xei", "Xeg", "K", "CI", "ISA"]
reserved_cols = ["Q_obs", "date", "ID"] + exclude_vars
VIF_THRESHOLD = 10.0

# ---------- pretty_name function ----------
def pretty_name(feat):
    mapping = {
        "Sproxy": r"$\mathrm{S}_{\mathrm{proxy}}$",
        "S_proxy": r"$\mathrm{S}_{\mathrm{proxy}}$",
        "Q_XAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QGXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "Pt-1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "Pt_1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "Pt": r"$\mathrm{P}_{\mathrm{t}}$",
        "P": r"$\mathrm{P}_{\mathrm{t}}$",
        "SM1": r"$\mathrm{SM}_{1}$",
        "SM2": r"$\mathrm{SM}_{2}$",
        "SM3": r"$\mathrm{SM}_{3}$",
        "SM4": r"$\mathrm{SM}_{4}$",
        "t": r"$\mathrm{t}$",
        "tmin": r"$\mathrm{t}_{\mathrm{min}}$",
        "tmax": r"$\mathrm{t}_{\mathrm{max}}$",
        "tmean": r"$\mathrm{t}_{\mathrm{mean}}$",
        "Td": r"$\mathrm{T}_{\mathrm{d}}$",
        "ET": r"$\mathrm{ET}$",
        "Rn": r"$\mathrm{R}_{\mathrm{n}}$",
        "RH": r"$\mathrm{RH}$",
        "NDVI": r"$\mathrm{NDVI}$",
        "LST": r"$\mathrm{LST}$",
        "Wind": r"$\mathrm{Wind}$",
        "Solar": r"$\mathrm{Solar}$",
        "Thermal": r"$\mathrm{Thermal}$",
        "Area": r"$\mathrm{Area}$",
        "SoilC": r"$\mathrm{SoilC}$",
        "SoilD": r"$\mathrm{SoilD}$",
        "LLM": r"$\mathrm{LLM}$",
        "LUM": r"$\mathrm{LUM}$",
        "CG": r"$\mathrm{CG}$",
        "Ep": r"$\mathrm{E}_{\mathrm{p}}$",
        "LU(cropland)": r"$\mathrm{LU}(\mathrm{cropland})$",
        "LU(forest)": r"$\mathrm{LU}(\mathrm{forest})$",
        "LU(grass)": r"$\mathrm{LU}(\mathrm{grass})$",
        "LU(snow)": r"$\mathrm{LU}(\mathrm{snow})$",
        "DEM_ave": r"$\mathrm{DEM}_{\mathrm{ave}}$",
        "DEM_range": r"$\mathrm{DEM}_{\mathrm{range}}$",
        "Slope_ave": r"$\mathrm{Slope}_{\mathrm{ave}}$",
        "TWI_ave": r"$\mathrm{TWI}_{\mathrm{ave}}$",
    }
    if feat in mapping:
        return mapping[feat]
    if "_" in feat:
        return feat.replace("_", r"\_")
    return r"$\mathrm{%s}$" % feat

# ---------- plotting font config ----------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

TITLE_FS = 28
LABEL_FS = 24
TICK_FS = 18
CB_TICK_FS = 16

# ---------- load data ----------
df = pd.read_csv(DATA_CSV)
features = [c for c in df.columns if c not in reserved_cols]
X = df[features].select_dtypes(include=[np.number]).copy()

# remove constant & NaN
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print("Dropping constant columns:", const_cols)
    X = X.drop(columns=const_cols)
X_clean = X.dropna().copy()
print("Clean data shape for analysis:", X_clean.shape)

# ======================================================================
# 1. Pearson correlation heatmap（下三角）
# ======================================================================
pearson_corr = X_clean.corr(method='pearson')
pretty_labels = [pretty_name(c) for c in pearson_corr.columns]
mask = np.triu(np.ones_like(pearson_corr), k=1)

# --------⬇⬇ 关键：手动创建两个子图（主图 + 颜色条）⬇⬇--------
fig = plt.figure(figsize=(12, 14.6))
gs = fig.add_gridspec(
    2, 1,
    height_ratios=[20, 0.6],   
    hspace=0.25                # 主图与颜色条之间的距离
)

ax_p = fig.add_subplot(gs[0, 0])     # 主图
ax_cbar = fig.add_subplot(gs[1, 0])  # 专用颜色条区域

# -------- 绘制 heatmap（无自动 colorbar）--------
hm_p = sns.heatmap(
    pearson_corr,
    mask=mask,
    cmap="RdBu_r",
    vmin=-1, vmax=1, center=0,
    annot=False,
    square=True,
    linewidths=0.6,
    xticklabels=pretty_labels,
    yticklabels=pretty_labels,
    cbar=False,     # ← 很重要：不要让 seaborn 自动画颜色条
    ax=ax_p
)

# -------- 单独绘制颜色条，精确控制高度和位置 --------
norm = plt.Normalize(-1, 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
cbar = plt.colorbar(
    sm,
    cax=ax_cbar,
    orientation="horizontal"
)
cbar.ax.tick_params(labelsize=CB_TICK_FS)
cbar.outline.set_edgecolor("black")
cbar.outline.set_linewidth(1)

# -------- 主图的字体等 --------
ax_p.set_title("(a) Pearson Correlation Matrix", fontsize=TITLE_FS, fontweight='bold')
ax_p.set_xticklabels(pretty_labels, rotation=70, ha="right", fontsize=TICK_FS)
ax_p.set_yticklabels(pretty_labels, rotation=0, fontsize=TICK_FS)

# -------- 保存 --------
fig.savefig(os.path.join(OUT_DIR, "pearson_corr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 2. Spearman correlation heatmap（下三角）
# ======================================================================
spearman_corr, _ = spearmanr(X_clean)
spearman_df = pd.DataFrame(spearman_corr, columns=X_clean.columns, index=X_clean.columns)
spearman_df.to_csv(os.path.join(OUT_DIR, "spearman_corr.csv"))
pretty_labels_sp = [pretty_name(c) for c in spearman_df.columns]
mask_sp = np.triu(np.ones_like(spearman_df), k=1)

fig_s, ax_s = plt.subplots(figsize=(10, 12))
hm_s = sns.heatmap(
    spearman_df,
    mask=mask_sp,
    cmap="RdBu_r",
    vmin=-1, vmax=1, center=0,
    annot=False,
    square=True,
    linewidths=0.6,
    xticklabels=pretty_labels_sp,
    yticklabels=pretty_labels_sp,
    cbar_kws={"orientation": "horizontal", "pad": 0.08, "shrink": 0.7},
    ax=ax_s
)

# colorbar 改窄 + 黑框
cbar2 = hm_s.collections[0].colorbar
cbar2.ax.tick_params(labelsize=CB_TICK_FS)
cbar2.outline.set_edgecolor("black")
cbar2.outline.set_linewidth(1.3)

ax_s.set_title("(b) Spearman Correlation Matrix", fontsize=TITLE_FS, fontweight='bold')
ax_s.set_xticklabels(pretty_labels_sp, rotation=70, ha="right", fontsize=TICK_FS)
ax_s.set_yticklabels(pretty_labels_sp, rotation=0, fontsize=TICK_FS)

plt.tight_layout()
fig_s.savefig(os.path.join(OUT_DIR, "spearman_corr.png"), dpi=300)
plt.close(fig_s)

# ======================================================================
# 3. 计算 VIF & 保存表
# ======================================================================
X_vif = X_clean.copy()
vif_data = []
cols_vif = X_vif.columns.tolist()
for i, col in enumerate(cols_vif):
    try:
        vif_val = variance_inflation_factor(X_vif.values, i)
    except Exception as e:
        print(f"Error computing VIF for {col}: {e}")
        vif_val = np.nan
    vif_data.append({"feature": col, "pretty_label": pretty_name(col), "VIF": float(vif_val)})

vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
vif_df.to_csv(os.path.join(OUT_DIR, "VIF_table.csv"), index=False)

# 打印高 VIF 项
high_vif = vif_df[vif_df["VIF"] >= VIF_THRESHOLD]
if not high_vif.empty:
    print("Features with VIF >= ", VIF_THRESHOLD)
    print(high_vif[["feature", "VIF"]])
else:
    print("No feature with VIF >=", VIF_THRESHOLD)

# ======================================================================
# 4. VIF 柱状图（前8高 + 后5低）
# ======================================================================
# 只用于画图的数据：去掉无效/非正 VIF
plot_base = vif_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["VIF"])
plot_base = plot_base[plot_base["VIF"] > 0]

top8_vif = plot_base.nlargest(8, "VIF").copy()
bottom5_vif = plot_base.nsmallest(5, "VIF").copy()

top8_vif["type"] = "Static Features (High VIF)"
bottom5_vif["type"] = "Dynamic Features (Low VIF)"
plot_df = pd.concat([top8_vif, bottom5_vif], ignore_index=True)

fig_vif, ax_vif = plt.subplots(figsize=(6, 5))
colors = {
    "Static Features (High VIF)": "#2E86AB",
    "Dynamic Features (Low VIF)": "#F24236"
}

# 按类型画水平柱状图
for feat_type in plot_df["type"].unique():
    subset = plot_df[plot_df["type"] == feat_type]
    ax_vif.barh(
        y=subset["pretty_label"],
        width=subset["VIF"],
        color=colors[feat_type],
        label=feat_type,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.0
    )

max_vif = np.nanmax(plot_df["VIF"])
min_vif = np.nanmin(plot_df["VIF"])

# log 轴 & x 范围：只要比最大值稍大即可
ax_vif.set_xscale("log")
ax_vif.set_xlim(left=1, right=max_vif * 5.0)

# 数值标签：放在柱子内部靠右，用 pretty_label 做 y 坐标
# 数值标签：蓝色条在条内，红色条在条外右侧
for _, row in plot_df.iterrows():
    vif_val = row["VIF"]
    if np.isinf(vif_val):
        vif_str = "Inf"
    else:
        vif_str = f"{vif_val:.1e}" if vif_val > 1000 else f"{vif_val:.2f}"

    if row["type"] == "Static Features (High VIF)":
        # 蓝色：高 VIF，数值在条内靠右
        x_pos = vif_val / 1.5
        ha = "right"
    else:
        # 红色：低 VIF，数值在条外右侧
        x_pos = vif_val * 1.1
        ha = "left"

    ax_vif.text(
        x_pos,
        row["pretty_label"],
        vif_str,
        ha=ha,
        va="center",
        fontsize=16,
        fontweight="bold"
    )                                                                                                                              
# 阈值线
ax_vif.axvline(x=VIF_THRESHOLD, color="red", linestyle="--",
               linewidth=1.0, label=f"VIF Threshold ({VIF_THRESHOLD})")

ax_vif.set_xlabel("Variance Inflation Factor (VIF)", fontsize=18, fontweight="bold")
ax_vif.set_ylabel("Feature", fontsize=18, fontweight="bold")
ax_vif.set_title("(b) VIF Comparison", fontsize=20, fontweight="bold", pad=15)
ax_vif.legend(fontsize=15, loc="upper right")
ax_vif.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.7)
ax_vif.tick_params(axis="y", labelsize=18)
ax_vif.tick_params(axis="x", labelsize=18)

plt.tight_layout()
vif_fig_path = os.path.join(OUT_DIR, "VIF_top8_bottom5_comparison.png")
fig_vif.savefig(
    vif_fig_path,
    dpi=300,
    transparent=True,   # ← 关键参数：开启透明背景
    bbox_inches="tight"
)
plt.close(fig_vif)
