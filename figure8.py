import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

# === 字体设置：Times 优先，mathtext 使用直立罗马体风格（回退到 serif 如无 Times） ===
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# === 颜色映射 ===
color_map = {
    "Arid": "#F16588",
    "Semi-arid": "#f3e46f",
    "Semi-humid": "#5EC5C5",
    "Humid": "#1f77b4"
}

# === user-provided denorm params for Fig.8 ===
max_vals = {
    "Area": 10910.0, "SM4": 0.385, "NDVI": 0.8194, "Pt-1": 221.15,
    "Q_XAJ": 2732.0,  "Sproxy": 0.746
}
min_vals = {
    "Area": 267.0, "SM4": 0.221, "NDVI": 0.1, "Pt-1": 0,
    "Q_XAJ": 0.0,  "Sproxy": 0
}
units = {
    "Area": "km²", "SM4": "m³/m³", "NDVI": "", "Pt-1": "mm",
    "Q_XAJ": "m³/s",  "Sproxy": ""
}

# === pretty_name: 将原始特征名转为 mathtext 下标形式（直立罗马体） ===
def pretty_name(feat):
    """
    Return a bold math-mode string for feature names with subscripts where appropriate.
    This uses \\mathbf to render the math characters in bold.
    """
    mapping = {
        # SM
        "SM1": r"$\mathbf{SM_{1}}$",
        "SM2": r"$\mathbf{SM_{2}}$",
        "SM3": r"$\mathbf{SM_{3}}$",
        "SM4": r"$\mathbf{SM_{4}}$",
        # temperature variants
        "t": r"$\mathbf{t}$",
        "tmin": r"$\mathbf{t_{min}}$",
        "tmax": r"$\mathbf{t_{max}}$",
        "tmean": r"$\mathbf{t_{mean}}$",
        # precipitation/time
        "Pt": r"$\mathbf{P_{t}}$",
        "Pt-1": r"$\mathbf{P_{t-1}}$",
        "Pt_1": r"$\mathbf{P_{t-1}}$",
        "P": r"$\mathbf{P}$",
        # others
        "Rn": r"$\mathbf{R_{n}}$",
        "Td": r"$\mathbf{T_{d}}$",
        "DEM_ave": r"$\mathbf{DEM_{ave}}$",
        "DEM_range": r"$\mathbf{DEM_{range}}$",
        "Slope_ave": r"$\mathbf{Slope_{ave}}$",
        "TWI_ave": r"$\mathbf{TWI_{ave}}$",
        "Sproxy": r"$\mathbf{S_{proxy}}$",
        "Q_XAJ": r"$\mathbf{Q_{GXAJ}}$",
        "QXAJ": r"$\mathbf{Q_{GXAJ}}$",
        "QGXAJ": r"$\mathbf{Q_{GXAJ}}$",
        "NDVI": r"$\mathbf{NDVI}$",
        "LST": r"$\mathbf{LST}$",
        "Area": r"$\mathbf{Area}$",
        "AI": r"$\mathbf{AI}$",
        "ET": r"$\mathbf{ET}$",
        "RH": r"$\mathbf{RH}$",
        "Wind": r"$\mathbf{Wind}$",
        "Solar": r"$\mathbf{Solar}$",
        "Thermal": r"$\mathbf{Thermal}$",
        "SoilC": r"$\mathbf{SoilC}$",
        "SoilD": r"$\mathbf{SoilD}$",
        "LLM": r"$\mathbf{LLM}$",
        "LUM": r"$\mathbf{LUM}$",
        "CG": r"$\mathbf{CG}$",
    }
    if feat in mapping:
        return mapping[feat]
    # fallback for names with underscore: render left and right with subscripts
    if "_" in feat:
        left, right = feat.split("_", 1)
        return rf"$\mathbf{{{left}}}_{{\mathbf{{{right}}}}}$"
    return rf"$\mathbf{{{feat}}}$"

# ==== denorm helper ====
def denorm_series(arr, feat):
    """
    Convert normalized feature values to physical units using min_vals/max_vals.
    Assumes normalization was x_norm in [0,1] and original x = x_norm*(max-min)+min.
    If arr already appears outside [-0.05,1.05] range, we assume it's already denormalized and return as-is.
    """
    s = np.asarray(arr, dtype=float)
    if feat not in max_vals:
        return s  # no mapping available; return original

    maxv = float(max_vals[feat])
    minv = float(min_vals.get(feat, 0.0))

    # detect if likely normalized (most values inside [0,1])
    s_max = np.nanmax(s)
    s_min = np.nanmin(s)
    if (s_max <= 1.05 and s_min >= -0.05):
        # treat as normalized in [0,1]
        return s * (maxv - minv) + minv
    else:
        # already denormalized or different scaling -> return unchanged
        return s

def norm_from_denorm(x_denorm, feat):
    """inverse: normalized = (x - min)/(max - min) ; used for secondary axis"""
    maxv = float(max_vals.get(feat, np.nan))
    minv = float(min_vals.get(feat, 0.0))
    rng = (maxv - minv) if (maxv - minv) != 0 else 1.0
    return (x_denorm - minv) / rng

# === 数据读取与预处理 ===
df = pd.read_csv("LSTM_normalized.csv")
id_to_climate = {
    1: "Arid", 2: "Semi-humid", 3: "Semi-arid",
    4: "Semi-humid", 5: "Semi-humid", 6: "Semi-humid",
    7: "Humid", 8: "Humid", 9: "Humid"
}
df["ClimateZone"] = df["ID"].map(id_to_climate)

exclude_vars = ["Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg",
                "ISA", "LU(snow)", "K", "CI"]
X_all = df.drop(columns=["Q_obs", "date"] + exclude_vars)
y_all = df["Q_obs"]

# === 模型训练 ===
dtrain = xgb.DMatrix(X_all.drop(columns=["ID", "ClimateZone"]), label=y_all)
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
explainer = shap.TreeExplainer(model)

# === 图设置 ===
fig, axs = plt.subplots(4, 3, figsize=(14, 14))
axs = axs.flatten()

zone_list = ["Arid", "Semi-arid", "Semi-humid", "Humid"]
label_list = ["(a) Arid", "(b) Semi-arid", "(c) Semi-humid", "(d) Humid"]
subplot_idx = 0

# === 绘图（每个气候区取 Top-3 features 做 SHAP dependence） ===
for row_idx, zone in enumerate(zone_list):
    zone_df = df[df["ClimateZone"] == zone]
    X_zone = zone_df.drop(columns=["Q_obs", "date", "ID", "ClimateZone"] + exclude_vars)
    shap_values = explainer.shap_values(X_zone)

    # 获取 Top-3 特征
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_features = pd.DataFrame({
        "feature": X_zone.columns,
        "importance": mean_abs
    }).sort_values("importance", ascending=False).head(3)["feature"].tolist()

    # 每一行的左侧添加统一标签
    fig.text(0.03, 0.959 - 0.23 * row_idx, label_list[row_idx],
             fontsize=18, fontweight="bold", ha='left', va='center')

    for feature in top3_features:
        ax = axs[subplot_idx]
        feature_values = X_zone[feature].values
        shap_vals = shap_values[:, X_zone.columns.get_loc(feature)]

        # === 清洗数据：去除 NaN / NA / inf（更稳健，兼容不同 numpy/pandas 版本） ===
        fv_ser = pd.Series(feature_values)
        sv_ser = pd.Series(shap_vals)

        not_null_mask = (~fv_ser.isnull()) & (~sv_ser.isnull())

        finite_mask = np.ones(len(fv_ser), dtype=bool)
        try:
            finite_mask &= np.isfinite(fv_ser.values.astype(float))
        except Exception:
            pass
        try:
            finite_mask &= np.isfinite(sv_ser.values.astype(float))
        except Exception:
            pass

        mask = not_null_mask.values & finite_mask

        x_clean = feature_values[mask]
        y_clean = shap_vals[mask]

        # 如果清洗后样本太少，跳过绘图（避免报错）
        if len(x_clean) < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=12)
            subplot_idx += 1
            continue

        # === 反归一化处理（如果有 mapping） ===
        x_denorm = denorm_series(x_clean, feature)

        # Optional debug prints (uncomment to inspect ranges)
        # print(feature, "orig min/max:", np.nanmin(x_clean), np.nanmax(x_clean),
        #       "denorm min/max:", np.nanmin(x_denorm), np.nanmax(x_denorm))

        # 散点图（带透明度）
        ax.scatter(x_denorm, y_clean, color=color_map[zone], s=20, alpha=0.3)

        # === 判断是否适合使用 LOWESS 曲线 ===
        if (len(np.unique(x_denorm)) > 10) and (np.std(x_denorm) > 1e-6):
            sns.regplot(
                x=x_denorm, y=y_clean,
                scatter=False,
                ax=ax,
                color="red",
                line_kws={"linewidth": 2},
                lowess=True
            )
        else:
            sns.regplot(
                x=x_denorm, y=y_clean,
                scatter=False,
                ax=ax,
                color="red",
                line_kws={"linewidth": 2},
                lowess=False
            )

        # 将标题改为 pretty_name（mathtext，下标为直立罗马体）
        ax.set_title(pretty_name(feature), fontsize=20, fontweight="bold")

        # 横轴设置：仅显示反归一化的物理单位（删除顶部归一化坐标轴）
        if feature in units:
            unit = units.get(feature, "")
            if unit:
                ax.set_xlabel(f"{pretty_name(feature)} ({unit})", fontsize=18, fontweight="bold")
            else:
                ax.set_xlabel(f"{pretty_name(feature)}", fontsize=18, fontweight="bold")
        else:
            ax.set_xlabel("Feature value (denormalized if mapping available)", fontsize=18, fontweight="bold")

        ax.set_ylabel("SHAP Value", fontsize=18, fontweight="bold")

        subplot_idx += 1

# 删除多余子图（如果有）
for i in range(subplot_idx, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout(
    rect=[0.06, 0, 1, 1]
)
plt.savefig("Figure8_SHAP_dependence_12_subplots_denorm.png", dpi=500)
plt.close()