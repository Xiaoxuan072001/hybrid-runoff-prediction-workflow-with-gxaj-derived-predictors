import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
# === 全局字体设置：优先 Times New Roman，fallback to serif ===
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # mathtext uses Times-like glyphs
matplotlib.rcParams['axes.labelweight'] = 'bold'

# sizes
TITLE_FS = 16
LABEL_FS = 14
TICK_FS = 14

# === 读表 ===
df = pd.read_excel("XGBoost_outputs.xlsx", sheet_name="Test")

# 防 NaN：只对有效列计算min/max
cols_for_range = ["Q_obs", "Q_XAJ", "XGBoost"]
min_val = min(df[c].min() for c in cols_for_range if c in df.columns and df[c].dropna().size > 0)
max_val = max(df[c].max() for c in cols_for_range if c in df.columns and df[c].dropna().size > 0)

# === 画图（较小尺寸） ===
plt.figure(figsize=(5, 4))

# Q_XAJ (gold triangles)
plt.scatter(
    df["Q_obs"], df["Q_XAJ"],
    color="#6cc6d8", marker="^", alpha=0.8,
    label="GXAJ Model", edgecolors='none'
)

# RandomForest (blue circles)
plt.scatter(
    df["Q_obs"], df["XGBoost"],
    color="#ee7564", marker="o", alpha=0.6,
    label="XGBoost", edgecolors='none'
)

# y = x 对角线
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5)

# 标签与标题（Times 字体由 rcParams 控制）
plt.xlabel("Observed Flow (m³/s)", fontsize=LABEL_FS, fontweight='bold', family="Times New Roman")
plt.ylabel("Simulated Flow (m³/s)", fontsize=LABEL_FS, fontweight='bold', family="Times New Roman")
plt.title("(a) XGBoost", fontsize=TITLE_FS, fontweight='bold', family="Times New Roman")

plt.xticks(fontsize=TICK_FS, fontweight='bold', family="Times New Roman")
plt.yticks(fontsize=TICK_FS, fontweight='bold', family="Times New Roman")

# 图例
plt.legend(fontsize=14, frameon=False, loc="upper left")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("XGBoost_scatter_times.png", dpi=500)
plt.show()



# ====== 配置 ======
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # mathtext 尽量和 Times 风格一致
matplotlib.rcParams['axes.labelweight'] = 'bold'

TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 14
ANNOT_FS = 14

OUT_FN = "Figure_residuals_boxplot.png"
OUT_DPI = 600

# 文件与期望的预测列名映射（如果你文件中列名不一样，可在此修改）
file_model_pairs = [
    ("LightGBM_outputs.xlsx", "LightGBM"),
    ("CatBoost_outputs.xlsx", "CatBoost"),
    ("XGBoost_outputs.xlsx", "XGBoost"),
    ("RandomForest_outputs.xlsx", "RandomForest"),
]

sheet_name = "Test"   # 每个文件中的 sheet 名称

# ====== 评估函数 ======
def nse(y_true, y_pred):
    """Nash–Sutcliffe efficiency"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred)**2) / denom

# ====== 读取并组织数据 ======
rows = []
nse_dict = {}
for fname, model_col in file_model_pairs:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"文件未找到：{fname} (请确认路径或文件名)")
    df = pd.read_excel(fname, sheet_name=sheet_name)

    # 自动查找列名：优先使用映射的 model_col；若不存在则尝试自动识别
    if model_col in df.columns:
        pred_col = model_col
    else:
        # 可能模型列命名略有不同，尝试排除常见列后剩下的为候选预测列
        possible = [c for c in df.columns if c not in ("Q_obs", "date", "ID", "index")]
        if "Q_obs" not in df.columns:
            raise ValueError(f"{fname} 中找不到 'Q_obs' 列，请确认 sheet 内容格式。")
        # 候选中去掉 Q_obs
        possible = [c for c in possible if c != "Q_obs"]
        # 如果只有一个候选，采用它；否则尝试匹配部分字符串
        if len(possible) == 1:
            pred_col = possible[0]
        else:
            # 优先匹配包含 model_col 字样的列
            matches = [c for c in possible if model_col.lower() in c.lower()]
            if len(matches) == 1:
                pred_col = matches[0]
            else:
                # 兜底：如果存在名为 Model 或 XGBoost 等精确名则采纳第一个候选
                pred_col = possible[0]

    # 检查必要列
    if "Q_obs" not in df.columns:
        raise ValueError(f"{fname} 的 sheet {sheet_name} 中没有 'Q_obs' 列，无法继续。")
    # 取 obs 和 pred（并剔除 NaN）
    obs = df["Q_obs"].astype(float).values
    pred = df[pred_col].astype(float).values

    # 计算残差并构建长表
    valid_mask = np.isfinite(obs) & np.isfinite(pred)
    obs_v = obs[valid_mask]
    pred_v = pred[valid_mask]
    res_v = pred_v - obs_v

    # 汇总行
    for o, p, r in zip(obs_v, pred_v, res_v):
        rows.append({"model": os.path.splitext(os.path.basename(fname))[0].split("_")[0],  # 用文件名前缀作为标签
                     "obs": o, "pred": p, "residual": r})

    # 计算并存储 NSE（在有效样本上）
    nse_val = nse(obs_v, pred_v)
    nse_dict[os.path.splitext(os.path.basename(fname))[0].split("_")[0]] = nse_val

# 长表 DataFrame
res_df = pd.DataFrame(rows)

# 若模型顺序需要固定，指定 order（与 file_model_pairs 保持一致）
model_order = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f, _ in file_model_pairs]

# ====== 绘图：残差箱线图（箱 + 小抖动点） ======
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

ax = sns.boxplot(data=res_df, x="model", y="residual", order=model_order,
                 palette=["#b2d0e8", "#97ceff", "#5293c9", "#4e94bf"], showfliers=False,
                 width=0.6, boxprops={'linewidth':1.2})

# 在箱图上叠加少量抖动点显示分布与离群点
sns.stripplot(data=res_df, x="model", y="residual", order=model_order,
              jitter=0.18, size=4, alpha=0.4, color="k", dodge=False)

# 添加中位数与均值标记（均值为白点）
means = res_df.groupby("model")["residual"].mean().reindex(model_order)
medians = res_df.groupby("model")["residual"].median().reindex(model_order)
x_positions = np.arange(len(model_order))
for xi, m in zip(x_positions, means):
    ax.plot(xi, m, marker='D', color='white', markersize=8, markeredgecolor='black', markeredgewidth=1.0)

# 横线 y=0
ax.axhline(0, color='#df81a5', linestyle='--', linewidth=1.0)

# 标注每个模型的 NSE（显示在各箱子上方）
ymax = res_df["residual"].quantile(0.98)
y_annot = ymax + 0.05 * (res_df["residual"].max() - res_df["residual"].min() + 1e-8)
for xi, mname in enumerate(model_order):
    nse_val = nse_dict.get(mname, np.nan)
    ann_txt = f"NSE: {nse_val:.3f}" if np.isfinite(nse_val) else "NSE: n/a"
    ax.text(xi, y_annot, ann_txt, fontsize=ANNOT_FS, fontweight='bold',
            ha='center', va='bottom', family='Times New Roman')

# 美化与标签
ax.set_xlabel("")
ax.set_ylabel("Simulated − Observed (m³/s)", fontsize=LABEL_FS, fontweight='bold', family='Times New Roman')
ax.set_title("Residual distribution by model (Test set)", fontsize=TITLE_FS, fontweight='bold', family='Times New Roman')

ax.tick_params(axis='x', labelsize=TICK_FS)
ax.tick_params(axis='y', labelsize=TICK_FS)

plt.tight_layout()
plt.savefig(OUT_FN, dpi=OUT_DPI, bbox_inches='tight')
plt.show()

print(f"[完成] 残差箱线图已保存为：{OUT_FN}")
