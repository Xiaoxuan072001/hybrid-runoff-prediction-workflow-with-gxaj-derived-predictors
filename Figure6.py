import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# === 全局字体和尺寸设置（Times 优先，mathtext 用直立 Roman） ===
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 26
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
# mathtext 设置以尽量匹配 Times 风格且使用直立罗马体
matplotlib.rcParams['mathtext.fontset'] = 'stix'    # good Times-like math fonts
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# === 固定特征颜色映射（按你的要求） ===
feature_palette = {
    "P":      "#4e94bf",  # 蓝
    "Pt-1":   "#6cced9",  # 浅蓝
    "NDVI":   "#006400",  # 墨绿

    "Q_XAJ":  "#af89d4",  # 紫
    "CG":     "#cfa9e9",  # 浅紫
    "LUM":    "#f2dff5",  # 灰紫

    "LST":    "#de9149",  # 粉橙
    "Area":   "#eda3b2",  # 粉色
    "Td":     "#ffd700",  # 黄色
    "Sproxy": "#fff2b2",  # 淡黄色

    # SM1-4 渐变绿（从深到浅）
    "SM1":    "#5a986b",
    "SM2":    "#77b286",
    "SM3":    "#afe2b7",
    "SM4":    "#ccf2d0",
}

# === pretty_name: 将原始特征名转为 mathtext 下标形式（直立罗马体） ===
def pretty_name(feat):
    """
    返回一个 matplotlib mathtext 字符串，呈现为直立罗马体并带下标（如需）。
    不在映射表的 '_' 会被保留为原名（若需更多自动化请告知）。
    """
    mapping = {
        # Explicit mappings requested
        "SM1": r"$\mathrm{SM}_{1}$",
        "SM2": r"$\mathrm{SM}_{2}$",
        "SM3": r"$\mathrm{SM}_{3}$",
        "SM4": r"$\mathrm{SM}_{4}$",
        "tmin": r"$\mathrm{t}_{\mathrm{min}}$",
        "tmax": r"$\mathrm{t}_{\mathrm{max}}$",
        "tmean": r"$\mathrm{t}_{\mathrm{mean}}$",
        "Rn": r"$\mathrm{R}_{\mathrm{n}}$",
        "Pt": r"$\mathrm{P}_{\mathrm{t}}$",
        "Pt-1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "Pt_1": r"$\mathrm{P}_{\mathrm{t-1}}$",
        "P": r"$\mathrm{P}$",
        # DEM / Slope / TWI with underscore -> subscript of the suffix
        "DEM_ave": r"$\mathrm{DEM}_{\mathrm{ave}}$",
        "DEM_range": r"$\mathrm{DEM}_{\mathrm{range}}$",
        "Slope_ave": r"$\mathrm{Slope}_{\mathrm{ave}}$",
        "TWI_ave": r"$\mathrm{TWI}_{\mathrm{ave}}$",
        # common others
        "Sproxy": r"$\mathrm{S}_{\mathrm{proxy}}$",
        "Q_XAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "QGXAJ": r"$\mathrm{Q}_{\mathrm{GXAJ}}$",
        "NDVI": r"$\mathrm{NDVI}$",
        "LST": r"$\mathrm{LST}$",
        "Area": r"$\mathrm{Area}$",
        "Td": r"$\mathrm{T}_{\mathrm{d}}$",
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
        "LU(cropland)": r"$\mathrm{LU}(\mathrm{cropland})$",
        "LU(forest)": r"$\mathrm{LU}(\mathrm{forest})$",
        "LU(grass)": r"$\mathrm{LU}(\mathrm{grass})$",
        "LU(snow)": r"$\mathrm{LU}(\mathrm{snow})$",
    }
    if feat in mapping:
        return mapping[feat]
    # fallback: if contains underscore, render left part as roman and right part as subscript roman
    if "_" in feat:
        left, right = feat.split("_", 1)
        return rf"$\mathrm{{{left}}}_{{\mathrm{{{right}}}}}$"
    # default: upright roman without subscript
    return rf"$\mathrm{{{feat}}}$"

# === 数据加载与处理 ===
df = pd.read_csv("LSTM_normalized.csv")
id_to_climate = {
    1: "Arid", 2: "Semi-humid", 3: "Semi-arid",
    4: "Semi-humid", 5: "Semi-humid", 6: "Semi-humid",
    7: "Humid", 8: "Humid", 9: "Humid"
}
df["ClimateZone"] = df["ID"].map(id_to_climate)
exclude_vars = ["Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg",
                "ISA", "LU(snow)", "K", "CI"]
# 构造特征矩阵（保留 ID 和 ClimateZone 以便后续分区）
X_all = df.drop(columns=["Q_obs", "date"] + exclude_vars)
y_all = df["Q_obs"]

# 训练 XGBoost（使用全部样本训练以计算 SHAP）
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

# === 图设置（A4 适配） ===
climate_zones = ["Arid", "Semi-arid", "Semi-humid", "Humid"]
fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
axs = axs.flatten()

# 子图编号前缀
prefixes = ['(a)', '(b)', '(c)', '(d)']

for idx, zone in enumerate(climate_zones):
    zone_df = df[df["ClimateZone"] == zone].copy()
    X_zone = zone_df.drop(columns=["Q_obs", "date", "ID", "ClimateZone"] + exclude_vars)
    # 计算该子集的 SHAP 值（注意 explainer 基于全量模型）
    shap_values = explainer.shap_values(X_zone)

    mean_abs_shap = pd.DataFrame({
        "feature": X_zone.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False).head(10).reset_index(drop=True)

    # 保护：若出现未设置颜色的特征，提示补色
    missing = set(mean_abs_shap["feature"]) - set(feature_palette)
    if missing:
        raise ValueError(f"这些特征未指定颜色，请在 feature_palette 中补充：{sorted(missing)}")

    # 绘图（水平条形）
    sns.barplot(
        data=mean_abs_shap,
        x="importance", y="feature",
        hue="feature",
        palette=feature_palette,
        ax=axs[idx],
        dodge=False,
        legend=False,
        edgecolor="black",
        linewidth=1.8,
        saturation=1,
        orient="h"
    )

    # annotate numeric values at bar ends
    max_val = mean_abs_shap["importance"].max()
    for i, (feature, val) in enumerate(zip(mean_abs_shap["feature"], mean_abs_shap["importance"])):
        axs[idx].text(val + 0.02 * max_val, i, f"{val:.2f}",
                      va="center", fontsize=18, fontweight="bold", zorder=10,
                      family='serif')

    # set title and axis labels
    axs[idx].set_title(f"{prefixes[idx]} {zone} zone", fontsize=26, fontweight="bold")
    axs[idx].set_xlabel("Mean absolute SHAP value", fontsize=22, fontweight="bold")
    axs[idx].set_ylabel("Feature", fontsize=22, fontweight="bold")

    # replace ytick labels by pretty (mathtext) names (rendered as upright roman)
    ytick_labels = [pretty_name(f) for f in mean_abs_shap["feature"].tolist()]
    axs[idx].set_yticks(range(len(ytick_labels)))
    axs[idx].set_yticklabels(ytick_labels, fontsize=20)

    axs[idx].tick_params(axis='x', labelsize=20, width=2)
    axs[idx].tick_params(axis='y', labelsize=20, width=2)

    # make spines thicker
    for spine in axs[idx].spines.values():
        spine.set_linewidth(2)

    axs[idx].set_xlim(0, max_val * 1.25)

# global layout & save
plt.tight_layout()
plt.savefig("Figure6_Top10_SHAP_by_ClimateZone_times_font.png", dpi=600)
plt.close()
