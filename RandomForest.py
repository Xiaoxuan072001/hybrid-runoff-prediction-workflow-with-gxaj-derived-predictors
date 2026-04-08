import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
# ========== 配置 ==========
output_dir = "SHAP-RandomForest"
os.makedirs(output_dir, exist_ok=True)

# === 数据读取 ===
df = pd.read_csv("LSTM_normalized.csv")
static_vars = ["DEM_ave", "DEM_range", "Slope_ave", "TWI_ave", "LU(cropland)", "LU(forest)", "LU(grass)", "Area", "SoilC", "SoilD", "LLM", "LUM", "CG"]
dynamic_vars = ["Q_XAJ", "P", "Pt-1", "NDVI", "Wind", "SM1", "SM2", "SM3", "SM4", "LST", "t", "tmin", "tmax", "Td", "ET", "Rn", "RH", "Solar", "Thermal", "h", "AI", "Ep", "Sproxy"]
exclude_vars = [ "Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg", "ISA", "LU(snow)", "K", "CI"]

static_colors = ["#b2d0e8", "#97ceff", "#5293c9"]
dynamic_colors = ["#df81a5", "#e797b4", "#ecadc4"]
def get_color(var): return static_colors[hash(var)%3] if var in static_vars else dynamic_colors[hash(var)%3]

id_to_name = {
    1: "Changmabao",
    2: "Maduwang",
    3: "Fuping",
    4: "Mengyin",
    5: "Wangjiashaozhuang",
    6: "Shuimingya",
    7: "Anxi",
    8: "Tongdao",
    9: "Tunxi"
}
# === 流域编号与颜色映射 ===
id_to_color = {
    1: "#F16588",
    2: "#F7969F",
    3: "#f2def5",
    4: "#8f89bb",
    5: "#a7c5fb",
    6: "#f3e46f",
    7: "#9cccce",
    8: "#D6F0D7",
    9: "#5EC5C5"
}
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def kge(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r = np.corrcoef(y_pred, y_true)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = (np.std(y_pred) / np.mean(y_pred)) / (np.std(y_true) / np.mean(y_true))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

def pbias(y_true, y_pred):
    return 100.0 * np.sum(y_true - y_pred) / np.sum(y_true)

# === 模型训练 ===
X_all = df.drop(columns=["Q_obs", "ID", "date"])
X_all = X_all.drop(columns=[col for col in exclude_vars if col in X_all.columns])
y_all = df["Q_obs"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

train_start = time.time()
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_end = time.time()

# === 评估指标 ===
def add_metrics(prefix, y_true, y_pred):
    return {
        "Model": "RandomForest",
        "Set": prefix,
        "NSE": nse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "PBIAS": pbias(y_true, y_pred),
        "KGE": kge(y_true, y_pred)
    }

metrics = [
    add_metrics("Train", y_train, y_train_pred),
    add_metrics("Test", y_test, y_test_pred)
]

# === 构建输出结果 ===
train_df = df.loc[X_train.index, ["date", "Q_obs", "Q_XAJ"]].copy()
test_df = df.loc[X_test.index, ["date", "Q_obs", "Q_XAJ"]].copy()
train_df["RandomForest"] = y_train_pred
test_df["RandomForest"] = y_test_pred

# === 平均指标 ===
metrics_df = pd.DataFrame(metrics)
avg_row = metrics_df[["NSE", "RMSE", "PBIAS", "KGE"]].mean().to_dict()
avg_row.update({"Model": "RandomForest", "Set": "Average"})
metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)

# === 写入Excel ===
metrics_output = "RandomForest_outputs.xlsx"
with pd.ExcelWriter(metrics_output) as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    test_df.to_excel(writer, sheet_name="Test", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

print(f"✅ 成功保存评估结果至 {metrics_output}")

shap_start = time.time()
explainer = shap.TreeExplainer(model)

# === 各流域 SHAP 条形图 ===
fig, axes = plt.subplots(3, 3, figsize=(18, 20))
for idx, (basin_id, name) in enumerate(id_to_name.items()):
    if basin_id not in df["ID"].unique(): continue
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    group = df[df["ID"] == basin_id].copy()
    X = group.drop(columns=["Q_obs", "ID", "date"])
    X = X.drop(columns=[col for col in exclude_vars if col in X.columns])
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "Variable": X.columns,
        "SHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values("SHAP", ascending=True)
    shap_df["Color"] = shap_df["Variable"].apply(get_color)

    bars = ax.barh(shap_df["Variable"], shap_df["SHAP"], color=shap_df["Color"], edgecolor='black', linewidth=0.6)
    max_shap = shap_df["SHAP"].max()
    ax.set_xlim(0, max_shap * 1.1)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max_shap * 0.015, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', fontsize=9)

    ax.set_title(name, fontsize=14)
    ax.set_xlabel("Mean(|SHAP value|)", fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{output_dir}/ALL_indiv_barplots_3x3.png", dpi=300)
plt.close()

# === SHAP summary 图 ===
image_paths = []
for basin_id, group in df.groupby("ID"):
    if basin_id not in id_to_name: continue
    name = id_to_name[basin_id]
    print(f"[生成 SHAP summary：{name}]")

    X = group.drop(columns=["Q_obs", "ID", "date"])
    X = X.drop(columns=[col for col in exclude_vars if col in X.columns])
    shap_values = explainer.shap_values(X)

    low, high = np.percentile(shap_values, [1, 99])
    shap_values_clipped = np.clip(shap_values, low, high)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_shap", ["#BFE9FE", "#72C2FF", "#7087E4", "#191A94"])
    shap.summary_plot(shap_values_clipped, X, show=False, cmap=cmap)
    plt.title(f"{name}", fontsize=18)
    plt.xlabel("SHAP Value", fontsize=18)
    plt.ylabel("Feature", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-30, 60)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    save_path = f"{output_dir}/tmp_summary_{name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    image_paths.append(save_path)

# 拼图
images = [Image.open(p) for p in image_paths]
w, h = images[0].size
grid_img = Image.new("RGB", (w * 3, h * 3), (255, 255, 255))
for i, img in enumerate(images):
    row, col = divmod(i, 3)
    grid_img.paste(img, (col * w, row * h))
grid_img.save(f"{output_dir}/ALL_summary_3x3.png")
print("[完成] 拼图保存成功：ALL_summary_3x3.png")

# 散点图
shap_values_all = explainer.shap_values(X_all)
colors = df["ID"].map(id_to_color)
log_transform_vars = ["Q_XAJ", "P", "Pt0"]

def plot_shap_scatter_grid(var_list, shap_values_all, X_all, colors,
                           id_to_name, id_to_color, filename,
                           n_cols=4, n_rows=None, add_trend=False):

    if n_rows is None:
        n_rows = int(np.ceil((len(var_list) + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    markers = {1: 'o', 4: 'o', 7: 'o', 2: 's', 5: 's', 8: 's', 3: '^', 6: '^', 9: '^'}

    for i, feature in enumerate(var_list):
        ax = axes[i]
        for basin_id in df["ID"].unique():
            if basin_id not in markers: continue
            marker = markers[basin_id]
            subset = df[df["ID"] == basin_id]
            if len(subset) > 150:
                subset = subset.sample(150, random_state=42)

            feature_vals = subset[feature].values.astype(float)
            jitter_strength = 0.015 * (np.max(feature_vals) - np.min(feature_vals))
            feature_vals += np.random.normal(0, jitter_strength, size=len(feature_vals))

            shap_raw = shap_values_all[subset.index, X_all.columns.get_loc(feature)]
            shap_plot_vals = np.sign(shap_raw) * np.log1p(np.abs(shap_raw)) if feature in log_transform_vars else shap_raw

            ax.scatter(feature_vals, shap_plot_vals, c=subset["ID"].map(id_to_color), s=18, alpha=0.7,
                       edgecolor='none', marker=marker)

        if add_trend:
            sorted_idx = np.argsort(feature_vals)
            x_sorted = feature_vals[sorted_idx]
            y_sorted = shap_plot_vals[sorted_idx]
            smoothed = lowess(y_sorted, x_sorted, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=1.2, alpha=0.9, linestyle='-')

        ax.set_title(feature, fontsize=20)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel("log(SHAP value)" if feature in log_transform_vars else "SHAP value", fontsize=16)
        ax.set_xlabel("Feature Value", fontsize=16)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=id_to_name[i],
                              markerfacecolor=color, markersize=10, markeredgecolor='none')
                       for i, color in id_to_color.items() if i in markers and i in id_to_name]
    legend_ax.legend(handles=legend_elements, loc="center", frameon=False,
                     ncol=1 if len(legend_elements) <= 9 else 2,
                     handletextpad=0.5, columnspacing=1.5, fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[完成] 已保存 SHAP 子图图像：{filename}")

plot_shap_scatter_grid(static_vars, shap_values_all, X_all, colors,
                       id_to_name, id_to_color, "shap_static_vars.png",
                       n_cols=4, n_rows=4, add_trend=False)

plot_shap_scatter_grid(dynamic_vars, shap_values_all, X_all, colors,
                       id_to_name, id_to_color, "shap_dynamic_vars.png",
                       n_cols=4, n_rows=None, add_trend=True)  # ✅ 使用 None 自动推断


shap_end = time.time()
print(f"[SHAP 分析总用时] {shap_end - shap_start:.2f} 秒")
