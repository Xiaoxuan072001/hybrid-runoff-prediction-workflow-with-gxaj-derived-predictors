# figure9_plot.py
# 读取四个模型的 CSV（feature, shap, pfi），绘制 Figure 9：
# (a) Top-k overlap across models；(b) SHAP–PFI Spearman rank correlation.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.ticker import PercentFormatter

# ===== 配置 =====
INPUT_DIR = "Figure11_outputs"   # 放四个 csv 的文件夹
FILE_MAP = {
    "XGBoost":       "fi_XGBoost.csv",
    "LightGBM":      "fi_LightGBM.csv",
    "Random Forest": "fi_RandomForest.csv",
    "CatBoost":      "fi_CatBoost.csv",
}
# 显示的简称
SHORT_NAMES = ["XGB", "LGB", "RF", "CB"]
TOP_K = 5
SAVE_PATH = os.path.join(INPUT_DIR, "Figure11_CrossModel_Consistency_pubstyle.png")

# 全局字体（Times New Roman + Bold，字号大一些）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# ===== 读取数据 =====
model_names_full = list(FILE_MAP.keys())
shap_rank = {}   # 每模型按 SHAP 从大到小排序的 Series
pfi_all   = {}   # 每模型 PFI Series
topk_sets = {}

for name, fname in FILE_MAP.items():
    path = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 的 CSV 未找到：{path}")
    df = pd.read_csv(path, index_col=0)
    if "shap" not in df.columns or "pfi" not in df.columns:
        raise ValueError(f"{path} 需要包含列 'shap' 与 'pfi'")

    shap_s = df["shap"].astype(float).sort_values(ascending=False)
    pfi_s  = df["pfi"].astype(float).reindex(shap_s.index)

    shap_rank[name] = shap_s
    pfi_all[name]   = pfi_s
    topk_sets[name] = set(shap_s.head(TOP_K).index)

# ===== (a) Top-k 交集比例矩阵 |∩|/k =====
n = len(model_names_full)
overlap = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        A = topk_sets[model_names_full[i]]
        B = topk_sets[model_names_full[j]]
        overlap[i, j] = len(A & B) / TOP_K

# ===== (b) SHAP–PFI Spearman ρ（各模型在其 Top-k 上）=====
rho_vals, p_vals = [], []
for name in model_names_full:
    shap_topk = shap_rank[name].head(TOP_K)
    pfi_topk  = pfi_all[name].reindex(shap_topk.index)
    mask = ~(shap_topk.isna() | pfi_topk.isna())
    if mask.sum() >= 2:
        rho, p = spearmanr(
            shap_topk[mask].rank(ascending=False),
            pfi_topk[mask].rank(ascending=False)
        )
    else:
        rho, p = np.nan, np.nan
    rho_vals.append(rho)
    p_vals.append(p)

# ===== 画图 =====
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

# (a) 热力图（完整矩阵；标注百分比）
from matplotlib.colors import LinearSegmentedColormap

# ===== (a) 热力图（完整矩阵；标注百分比） =====
ax1 = fig.add_subplot(gs[0, 0])

# 自定义渐变 colormap
colors_a = ["#dbed9b", "#f0e8a0", "#fbc8b0", "#f7969f"]
cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", colors_a)

hm = sns.heatmap(
    overlap, ax=ax1,
    annot=True, fmt=".0%", vmin=0, vmax=1,
    cmap=cmap_custom,   # ✅ 使用自定义糖果渐变
    annot_kws={"fontsize": 16, "fontweight": "bold", "fontname": "Times New Roman"},
    cbar_kws={"label": f"Top-{TOP_K} overlap ratio"}
)

# 百分比刻度
cbar = hm.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel(f"Top-{TOP_K} overlap ratio", fontweight="bold", fontsize=15, fontname="Times New Roman")

ax1.set_xticklabels(SHORT_NAMES, rotation=0, ha="center", fontsize=15, fontweight="bold")
ax1.set_yticklabels(SHORT_NAMES, rotation=0, fontsize=15, fontweight="bold")
ax1.set_title(f"(a) Top-{TOP_K} overlap across models", pad=8, fontweight="bold", fontsize=18)

# (b) 条形图（加 margin，避免顶到上边框；显著性 *）
ax2 = fig.add_subplot(gs[0, 1])
bar_color = "#dbed9b"  
bars = ax2.bar(range(n), rho_vals, color=bar_color, edgecolor="black", linewidth=1.2)

ax2.set_xticks(range(n))
ax2.set_xticklabels(SHORT_NAMES, rotation=0, ha="center", fontsize=15, fontweight="bold")
y_upper = (np.nanmax(rho_vals) if len(rho_vals) else 1.0) + 0.15  # 顶部预留
y_upper = min(1.05, y_upper)  # 不超过 1.05
ax2.set_ylim(0, y_upper)
ax2.axhline(0, color="k", linewidth=1.0)
ax2.set_ylabel("Spearman ρ (SHAP vs PFI)", fontsize=15, fontweight="bold")
ax2.set_title(f"(b) SHAP–PFI rank correlation (Top-{TOP_K})", pad=8, fontweight="bold")

for b, rho, p in zip(bars, rho_vals, p_vals):
    if np.isnan(rho):
        label = "NA"
    else:
        star = "*" if (not np.isnan(p) and p < 0.05) else ""
        label = f"ρ={rho:.2f}{star}\np={p:.3f}"
    ax2.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 0.03,
        label, ha="center", va="bottom",
        fontsize=14, fontweight="bold"
    )

# 手动调整边距（替代 tight_layout，避免 warning）
plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.15, wspace=0.32)
plt.savefig(SAVE_PATH, dpi=600)
plt.close()
print(f"✅ Figure saved: {SAVE_PATH}")


'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ——— Models & tools ———
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap

# ====== Config ======
random_state = 42
top_k = 5
out_dir = "Figure7_outputs"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

# ====== Load data ======
df = pd.read_csv("LSTM_normalized.csv")
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

exclude_vars = ["Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg",
                "ISA", "LU(snow)", "K", "CI"]
drop_cols = [c for c in ["Q_obs", "date", "ID"] if c in df.columns] + exclude_vars
features = [c for c in df.columns if c not in drop_cols]
X = df[features].copy()
y = df["Q_obs"].copy()

# same split for all models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# ====== Define models with your hyperparameters ======
models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=0.5,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=1.0,
        random_state=random_state,
        n_jobs=-1,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3,
        subsample=0.7,
        verbose=0,
        random_seed=random_state,
    ),
}

# ====== Fit, compute SHAP & PFI, save CSV ======
shap_rank = {}        # pd.Series (importance) sorted desc
pfi_series = {}       # pd.Series (importance mean)
topk_sets = {}

for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)

    # SHAP (TreeExplainer works for all tree models)
    print(f"Computing SHAP for {name} ...")
    explainer = shap.TreeExplainer(model)
    # 用测试集做重要度（避免数据泄漏）
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):  # 某些模型可能返回多输出
        shap_vals = shap_vals[0]
    shap_importance = np.abs(shap_vals).mean(axis=0)

    shap_s = pd.Series(shap_importance, index=features, name="shap").sort_values(ascending=False)
    shap_rank[name] = shap_s

    # PFI（测试集，RMSE 方向）
    print(f"Computing PFI for {name} ...")
    pfi = permutation_importance(
        model, X_test, y_test,
        scoring="neg_mean_squared_error",
        n_repeats=10, random_state=random_state, n_jobs=-1
    )
    pfi_s = pd.Series(pfi.importances_mean, index=features, name="pfi")
    pfi_series[name] = pfi_s

    # 合并、保存
    imp_df = pd.concat([shap_s.rename("shap"), pfi_s.rename("pfi")], axis=1)
    imp_df.to_csv(os.path.join(out_dir, f"fi_{name.replace(' ', '')}.csv"), index_label="feature")

    # Top-k set（基于 SHAP 排名）
    topk_sets[name] = set(shap_s.head(top_k).index)

'''
