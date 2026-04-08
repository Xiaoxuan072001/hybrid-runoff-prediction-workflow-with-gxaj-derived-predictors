import os
import time
import pandas as pd
import numpy as np
import shap
import matplotlib.patches as mpatches
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as patheffects
# 配置
output_dir = "SHAP-XGBoost"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv("LSTM_normalized.csv")
df["date"] = pd.to_datetime(df["date"])

static_vars = ["DEM_ave", "DEM_range", "Slope_ave", "TWI_ave", "LU(cropland)", "LU(forest)", "LU(grass)", "Area", "SoilC", "SoilD", "LLM", "LUM", "CG", "LU(snow)"]
dynamic_vars = ["Q_XAJ", "P", "Pt-1", "NDVI", "Wind", "SM1", "SM2", "SM3", "SM4", "LST", "t", "tmin", "tmax", "Td", "ET", "Rn", "RH", "Solar", "Thermal", "h", "AI", "Ep", "Sproxy"]
exclude_vars = [ "Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg", "K", "CI", "ISA"]
max_vals = {
    "Area": 10910.0, "SM1": 0.4256, "NDVI": 0.8194, "Pt-1": 221.15, 
    "LST": 39.76, "DEM_range": 3690.0, "Q_XAJ": 2732.0, "Pt": 221.15, "ET": 10 
}
min_vals = {
    "Area": 313.0, "SM1": 0.137, "NDVI": 0.01, "Pt-1":0, 
    "LST": -4.06, "DEM_range": 343.0, "Q_XAJ": 0, "Pt": 0, "ET": 0 
}
units = {
    "Area": "km²", "SM2": "m³/m³", "SM1": "m³/m³", "NDVI": "", "Pt-1": "mm", "LST": "°C",
    "DEM_range": "m", "Q_XAJ": "m³/s", "Pt": "mm", "ET": "mm"
    }
static_colors = "#6cc6d8"
dynamic_colors = "#ee7564"
def get_color(var): return static_colors[hash(var)%3] if var in static_vars else dynamic_colors[hash(var)%3]

id_to_name = {
    1: "Changmabao", 2: "Maduwang", 3: "Fuping", 4: "Mengyin",
    5: "Wangjiashaozhuang", 6: "Shuimingya", 7: "Anxi", 8: "Tongdao", 9: "Tunxi"
}
id_to_color = {
    1: "#F16588", 2: "#F7969F", 3: "#f2def5", 4: "#8f89bb",
    5: "#a7c5fb", 6: "#f3e46f", 7: "#9cccce", 8: "#D6F0D7", 9: "#5EC5C5"
}

features = [col for col in df.columns if col not in ["Q_obs", "date", "ID"] + exclude_vars]
X_all = df[features]
y_all = df["Q_obs"]

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# 训练模型
train_start = time.time()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'lambda': 0.5,
    'alpha': 0.5,
    'seed': 42
}
model = xgb.train(params, dtrain, num_boost_round=300)
train_pred = model.predict(dtrain)
test_pred = model.predict(dtest)
train_end = time.time()

# 评估函数
def nse(y_true, y_pred): return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def pbias(y_true, y_pred): return 100.0 * np.sum(y_true - y_pred) / np.sum(y_true)
def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

# 保存预测和指标
train_df = df.loc[X_train.index, ["date", "Q_obs", "Q_XAJ"]].copy()
test_df = df.loc[X_test.index, ["date", "Q_obs", "Q_XAJ"]].copy()
train_df["XGBoost"] = train_pred
test_df["XGBoost"] = test_pred

metrics_df = pd.DataFrame([
    {"Model": "XGBoost", "Set": "Train", "NSE": nse(y_train, train_pred), "RMSE": rmse(y_train, train_pred),
     "PBIAS": pbias(y_train, train_pred), "KGE": kge(y_train, train_pred)},
    {"Model": "XGBoost", "Set": "Test", "NSE": nse(y_test, test_pred), "RMSE": rmse(y_test, test_pred),
     "PBIAS": pbias(y_test, test_pred), "KGE": kge(y_test, test_pred)},
])
avg_row = metrics_df[["NSE", "RMSE", "PBIAS", "KGE"]].mean().to_dict()
avg_row.update({"Model": "XGBoost", "Set": "Average"})
metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)

with pd.ExcelWriter("XGBoost_outputs.xlsx") as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    test_df.to_excel(writer, sheet_name="Test", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

print("✅ XGBoost 模型完成，结果已保存至 XGBoost_outputs.xlsx")
# ===========================
# pretty_name: 统一的标签转换（带下标，直立罗马体）
# ===========================
def pretty_name(feat):
    # explicit mappings for those you listed
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
        # DEM/Slope/TWI special transforms
        "DEM_ave": r"$\mathrm{DEM}_{\mathrm{ave}}$",
        "DEM_range": r"$\mathrm{DEM}_{\mathrm{range}}$",
        "Slope_ave": r"$\mathrm{Slope}_{\mathrm{ave}}$",
        "TWI_ave": r"$\mathrm{TWI}_{\mathrm{ave}}$",
    }
    if feat in mapping:
        return mapping[feat]

    # fallback: if contains underscore but not in mapping, escape underscore
    if "_" in feat:
        return feat.replace("_", r"\_")

    # default: render as upright roman
    return r"$\mathrm{%s}$" % feat



# ---------- 可调参数 ----------
zero_mask_rel = 0.01        # 排除 x≈0 附近的热力范围（占轴宽的比例）
density_point_scale = 260.0 # 热力点大小缩放
density_marker_square = True # True=方块格子, False=圆点
percent_max = 30.0          # top axis 百分比范围 0% - 30%
# ------------------------------------------------

# 自定义渐变色（使用你提供的色板并反转，使 low->high 对应 红->绿）
palette = ["#b4d789", "#DBED9B", "#E6E49F", "#F0E8A0", "#F4E198", "#F7D890",
           "#F8D388", "#FAD2A8", "#FBC8B0", "#F7969F", "#F16588"]
palette_rev = palette[::-1]
custom_cmap = LinearSegmentedColormap.from_list("custom_rdylgn", palette_rev, N=256)

# 依赖：model, X_all, pretty_name, output_dir 在脚本其他部分已定义
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_all)  # shape (n_samples, n_features)

# 全局字体/加粗
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# 排序（按 mean(|SHAP|) 降序）
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]
sorted_features = [X_all.columns[i] for i in sorted_idx]

# 裁剪 SHAP 值（避免极端值影响显示）
shap_clip = np.clip(shap_values[:, sorted_idx], -1500, 1500)
n_samples, n_feats = shap_clip.shape

# mean(|SHAP|) 与百分比（用于横条与注记）
mean_shap_sorted = np.abs(shap_clip).mean(axis=0)
total_mean = mean_shap_sorted.sum() if mean_shap_sorted.sum() > 0 else 1.0
percents = mean_shap_sorted / total_mean * 100.0

# 计算密度（2D histogram）
x_min_data = shap_clip.min()
x_max_data = shap_clip.max()
pad = max(1e-8, 0.02 * (x_max_data - x_min_data))
x_min = x_min_data - pad
x_max = x_max_data + pad

xbins = 200
x_edges = np.linspace(x_min, x_max, xbins + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_edges = np.arange(-0.5, n_feats + 0.5, 1.0)

x_list = []
y_list = []
for j in range(n_feats):
    x_vals = shap_clip[:, j]
    y_vals = np.full_like(x_vals, fill_value=j, dtype=float)
    x_list.append(x_vals)
    y_list.append(y_vals)
x_all = np.concatenate(x_list)
y_all = np.concatenate(y_list)

H, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges])
density = H.T  # shape (n_feats, xbins)
density_max = density.max() if density.max() > 0 else 1.0
density_norm = density / density_max
'''
# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(8.0, 8.0))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 内部横条（百分比映射到数据坐标）
axis_width = x_max - x_min
bar_left_margin = 0.02
bar_left = x_min + axis_width * bar_left_margin
bar_max_width = axis_width * 0.90

perc_clip = np.minimum(percents, percent_max)
bar_widths_data = (perc_clip / percent_max) * bar_max_width

desired_right = bar_left + bar_max_width
if desired_right > x_max:
    x_max = desired_right + 0.01 * axis_width
ax.set_xlim(x_min, x_max)

# 只画条轮廓（不填充），改为更浅的蓝色，线细一点
outline_color = "#aadce0"  # 更浅的蓝色
ax.barh(np.arange(n_feats), bar_widths_data, left=bar_left, height=0.40,
        color='none', edgecolor=outline_color, linewidth=0.8, zorder=1)

# 百分比注记：透明底、直接写上去（无描边、无白底）
for j in range(n_feats):
    txt_x = bar_left + 0.01 * bar_max_width
    ax.text(txt_x, j, f"({percents[j]:.1f}%)", va='center', ha='left',
            fontsize=12, fontweight='bold', zorder=7, color='black')

# 绘制 density 点，但排除 x≈0 附近（去掉中心实底）
zero_mask_abs = zero_mask_rel * (x_max - x_min)  # 以数据单位表示排除范围

for j in range(n_feats):
    row = density_norm[j, :]  # length xbins
    valid = row > 0
    if not np.any(valid):
        continue
    xs = x_centers[valid]
    # 过滤掉靠近 0 的格子
    keep_mask = np.abs(xs) >= zero_mask_abs
    if not np.any(keep_mask):
        continue
    xs_keep = xs[keep_mask]
    vals = row[valid][keep_mask]  # 0..1
    ys_keep = np.full(xs_keep.shape, j)
    sizes = vals * density_point_scale
    colors = custom_cmap(vals)
    if density_marker_square:
        ax.scatter(xs_keep, ys_keep, s=sizes, c=colors, marker='s', linewidths=0, alpha=0.9, zorder=0)
    else:
        ax.scatter(xs_keep, ys_keep, s=sizes, c=colors, marker='o', linewidths=0, alpha=0.9, zorder=0)

# SHAP beeswarm 点（按原始特征值着色），点在最上层
cmap = custom_cmap
X_sorted_vals = X_all.iloc[:, sorted_idx].values
norms = []
for j in range(n_feats):
    col = X_sorted_vals[:, j]
    vmin = np.nanmin(col)
    vmax = np.nanmax(col)
    if vmin == vmax:
        norms.append(Normalize(vmin - 1e-8, vmax + 1e-8))
    else:
        norms.append(Normalize(vmin, vmax))

rng = np.random.default_rng(42)
point_size = 18
for j in range(n_feats):
    shap_vals_j = shap_clip[:, j]
    y_jitter = rng.normal(loc=j, scale=0.12, size=n_samples)
    feat_vals = X_sorted_vals[:, j]
    colors = cmap(norms[j](feat_vals))
    ax.scatter(shap_vals_j, y_jitter, s=point_size, c=colors, linewidths=0, alpha=0.86, zorder=3)

# 轴与标签（加粗）
ytick_labels = [pretty_name(f) for f in sorted_features]
ax.set_yticks(np.arange(n_feats))
ax.set_yticklabels(ytick_labels, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.axvline(0.0, color='k', lw=0.6, zorder=2)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=12, fontweight='bold')

# top axis：百分比 0..percent_max
def data_to_percent(x):
    return (x - bar_left) / bar_max_width * percent_max
def percent_to_data(p):
    return bar_left + (p / percent_max) * bar_max_width

ax_top = ax.secondary_xaxis('top', functions=(data_to_percent, percent_to_data))
ax_top.set_xlabel("Mean(|SHAP|) percentage", fontsize=12, fontweight='bold')
ticks_percent = np.linspace(0, percent_max, 7)
ax_top.set_xticks(ticks_percent)
ax_top.set_xticklabels([f"{t:.0f}%" for t in ticks_percent], fontsize=11)
for lbl in ax_top.get_xticklabels():
    lbl.set_fontweight('bold')

# 主轴刻度加粗（兼容写法）
ax.tick_params(axis='x', labelsize=11)
for lbl in ax.get_xticklabels():
    lbl.set_fontweight('bold')
for lbl in ax.get_yticklabels():
    lbl.set_fontweight('bold')

# colorbar：紧邻右侧且较细，标签在右侧并加粗
fig.canvas.draw()
pos = ax.get_position()
cax_left = pos.x1 + 0.035
cax_bottom = pos.y0
cax_width = 0.018
cax_height = pos.height
cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
mappable_feat = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
mappable_feat.set_array([])
cbar = fig.colorbar(mappable_feat, cax=cax, orientation='vertical')
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')
cbar.set_label('Feature value', fontsize=14, fontweight='bold', labelpad=6)
cbar.set_ticks([0.0, 1.0])
cbar.set_ticklabels(['Low', 'High'])
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(12)
    t.set_fontweight('bold')

# 网格与边距
ax.grid(axis='x', linestyle=':', linewidth=0.6, color='#bdbdbd', zorder=0)
fig.subplots_adjust(left=0.12, right=0.88, top=0.96, bottom=0.03)

# 保存图像
out_path = os.path.join(output_dir, "figure5_shap_final_no_bbox_lightbar.png")
plt.savefig(out_path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved figure to {out_path}")


'''
# ----- 新：反归一化帮助函数（放在文件顶部，与 max_vals/min_vals/units 同级） -----
def denorm_series(arr, feat):
    """
    把归一化的 arr -> 物理量：x = x_norm * (max - min) + min
    如果 arr 明显超出 [0,1]（例如已是反归一化），则直接返回原数组。
    """
    s = np.asarray(arr, dtype=float)
    if feat not in max_vals:
        return s  # 没有 mapping 的特征，按原样返回

    maxv = float(max_vals[feat])
    minv = float(min_vals.get(feat, 0.0))
    rng = maxv - minv if (maxv - minv) != 0 else 1.0

    s_max = np.nanmax(s)
    s_min = np.nanmin(s)
    # 如果大部分值在 [0,1] 视为归一化过，进行反归一化；否则假设已经是原始物理量
    if (s_max <= 1.05) and (s_min >= -0.05):
        return s * rng + minv
    else:
        return s

# ----- 用下面这个函数替换原来的 plot_shap_selected8 -----
def plot_shap_selected8(shap_values, X_all, df, id_to_name, id_to_color, filename):
    """
    绘制 3x3 网格：前 8 个子图为指定变量的 SHAP dependence scatter (按流域着色/标记)，
    第9格作为图例区域。横轴使用反归一化后的物理量（如果 mapping 可用）。
    """
    # 校验 shap 与 X_all 维度一致
    if shap_values.shape[0] != X_all.shape[0] or shap_values.shape[1] != X_all.shape[1]:
        raise ValueError("shap_values 与 X_all 维度不匹配。请确认使用 explainer.shap_values(X_all)。")

    target_vars = ["Q_XAJ", "P", "Pt-1", "NDVI", "SM1", "ET", "Area", "LST"]
    var_list = [v for v in target_vars if v in X_all.columns]

    n_cols, n_rows = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    markers = {1: 'o', 4: 'o', 7: 'o', 2: 's', 5: 's', 8: 's', 3: '^', 6: '^', 9: '^'}

    for i, feature in enumerate(var_list[:8]):
        ax = axes[i]
        col_idx = X_all.columns.get_loc(feature)

        for basin_id in sorted(df["ID"].unique()):
            if basin_id not in markers:
                continue
            subset = df[df["ID"] == basin_id]
            if len(subset) == 0:
                continue
            if len(subset) > 150:
                subset = subset.sample(150, random_state=42)

            # 原始 x 值（可能是归一化或已反归一化）
            x_vals = subset[feature].astype(float).values
            # ---------- 关键：对横轴进行反归一化（若 mapping 存在） ----------
            x_denorm = denorm_series(x_vals, feature)

            # jitter 仍基于反归一化后的范围（更合理）
            if np.ptp(x_denorm) > 0:
                jitter = 0.01 * (np.nanmax(x_denorm) - np.nanmin(x_denorm))
                x_plot_vals = x_denorm + np.random.normal(0, jitter, size=len(x_denorm))
            else:
                x_plot_vals = x_denorm

            # shap 值（索引对齐）
            try:
                shap_vals = shap_values[subset.index, col_idx]
            except Exception:
                rel_idx = np.searchsorted(X_all.index.values, subset.index.values)
                shap_vals = shap_values[rel_idx, col_idx]

            ax.scatter(
                x_plot_vals, shap_vals,
                c=subset["ID"].map(id_to_color), marker=markers.get(basin_id, 'o'),
                s=20, alpha=0.7, edgecolor='none'
            )

        ax.axhline(0, color='black', linestyle='--', linewidth=1.2)

        # 横轴标签：使用 pretty_name + 单位（若有）
        unit = units.get(feature, "")
        if unit:
            ax.set_xlabel(f"{pretty_name(feature)} ({unit})", fontsize=18, fontweight="bold")
        else:
            ax.set_xlabel(pretty_name(feature), fontsize=18, fontweight="bold")

        ax.set_ylabel("SHAP value", fontsize=18, fontweight="bold")
        ax.tick_params(labelsize=16, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # 第9格作为图例
    legend_ax = axes[8]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], marker=markers[i], color='w', label=id_to_name[i],
               markerfacecolor=id_to_color[i], markersize=12)
        for i in sorted(id_to_name.keys())
    ]
    legend_ax.legend(handles=legend_handles, loc="center", frameon=False, ncol=1, fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[完成] 已保存 SHAP 8因子图：{out_path}")


# === 调用绘图函数：传入你已计算的 shap_values 变量 ===
plot_shap_selected8(
    shap_values,    # 这里使用你定义过的 shap_values（explainer.shap_values(X_all)）
    X_all, df, id_to_name, id_to_color,
    filename="Figure7_SHAP_selected8_XGB.png"
)

'''
# SHAP 散点图支持变量存在检查
def plot_shap_scatter_grid(var_list, shap_values_all, X_all, colors,
                           id_to_name, id_to_color, filename,
                           n_cols=4, n_rows=None, add_trend=False):

    var_list = [v for v in var_list if v in X_all.columns]  # ✅ 自动过滤无效变量
    total_axes_needed = len(var_list) + 1
    if n_rows is None:
        n_rows = int(np.ceil(total_axes_needed / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    if len(axes) > total_axes_needed:
        for ax in axes[total_axes_needed:]:
            ax.remove()
        axes = axes[:total_axes_needed]

    markers = {1: 'o', 4: 'o', 7: 'o', 2: 's', 5: 's', 8: 's', 3: '^', 6: '^', 9: '^'}

    for i, feature in enumerate(var_list):
        ax = axes[i]
        for basin_id in df["ID"].unique():
            if basin_id not in markers:
                continue
            subset = df[df["ID"] == basin_id]
            if len(subset) > 150:
                subset = subset.sample(150, random_state=42)

            feature_vals = subset[feature].values
            jitter = 0.015 * (np.max(feature_vals) - np.min(feature_vals))
            feature_vals += np.random.normal(0, jitter, len(feature_vals))

            shap_raw = shap_values_all[subset.index, X_all.columns.get_loc(feature)]
            shap_plot_vals = np.sign(shap_raw) * np.log1p(np.abs(shap_raw)) if feature in ["P", "Pt0"] else shap_raw

            ax.scatter(feature_vals, shap_plot_vals, c=subset["ID"].map(id_to_color), s=18, alpha=0.7,
                       edgecolor='none', marker=markers[basin_id])

        ax.set_title(feature, fontsize=20)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel("SHAP value", fontsize=14)
        ax.set_xlabel("Feature Value", fontsize=14)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_elements = [
        Line2D([0], [0], marker=markers[i], color='w', label=id_to_name[i],
               markerfacecolor=color, markersize=10)
        for i, color in id_to_color.items()
    ]
    legend_ax.legend(handles=legend_elements, loc="center", frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[完成] 已保存 SHAP 图：{filename}")

plot_shap_scatter_grid(static_vars, shap_values_all, X_all, df["ID"].map(id_to_color),
                       id_to_name, id_to_color, "shap_static_vars.png", n_cols=4)
plot_shap_scatter_grid(dynamic_vars, shap_values_all, X_all, df["ID"].map(id_to_color),
                       id_to_name, id_to_color, "shap_dynamic_vars.png", n_cols=4, add_trend=True)
'''