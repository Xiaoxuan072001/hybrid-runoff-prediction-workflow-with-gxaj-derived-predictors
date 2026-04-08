# fixed_full_script_catfirst_lgb_fix.py
import os
import time
import pandas as pd
import numpy as np
import shap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import multiprocessing

# ---------- psutil 安全导入（保证名称总存在） ----------
try:
    import psutil
except Exception:
    psutil = None

def get_sys_info():
    """Return simple string with memory and cpu usage (best-effort)."""
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            return f"CPU%={cpu:.1f}%, mem_used={mem.used//(1024**2)}MB/{mem.total//(1024**2)}MB ({mem.percent:.0f}%)"
        except Exception as e:
            return f"psutil installed but error: {e}"
    else:
        return "psutil not installed — system info unavailable"

# ---------- 配置 ----------
output_dir = "SHAP-A3"
os.makedirs(output_dir, exist_ok=True)

csv_path = "LSTM_normalized.csv"   # 若不在当前目录请修改路径
print("Loading CSV:", csv_path, flush=True)
df = pd.read_csv(csv_path)
print("CSV loaded — rows:", len(df), "cols:", len(df.columns), flush=True)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# 静态 / 动态变量（复用你给的）
static_vars = ["DEM_ave", "DEM_range", "Slope_ave", "TWI_ave", "LU(cropland)", "LU(forest)", "LU(grass)", "Area", "SoilC", "SoilD", "LLM", "LUM", "CG"]
dynamic_vars = ["Q_XAJ", "P", "Pt-1", "NDVI", "Wind", "SM1", "SM2", "SM3", "SM4", "LST", "t", "tmin", "tmax", "Td", "ET", "Rn", "RH", "Solar", "Thermal", "h", "AI", "Ep", "Sproxy"]
exclude_vars = [ "Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg", "ISA", "LU(snow)", "K", "CI"]


static_colors = "#6cc6d8"
dynamic_colors = "#ee7564"

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

# ========== 特征与目标 ==========
features = [col for col in df.columns if col not in ["Q_obs", "date", "ID"] + exclude_vars]
X_all = df[features].copy()
y_all = df["Q_obs"].copy()
print("Features used:", len(features), flush=True)

# 随机分割（与原脚本一致）
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
print("Train/test sizes:", X_train.shape, X_test.shape, flush=True)

# ========== 超参数（限制线程以避免占满）==========
cpu_count = multiprocessing.cpu_count()
thread_count_default = max(1, min(4, cpu_count - 1))
print("Detected CPU count:", cpu_count, "-> using up to", thread_count_default, "threads for boosted models", flush=True)

rf_total_trees = 200
rf_batch = 20  # 每次增加多少树（用于打印进度）

rf_params = {"n_estimators": rf_total_trees, "max_depth": 12, "min_samples_leaf": 1,
             "random_state": 42, "n_jobs": 1}

cat_params = {"iterations": 500, "learning_rate": 0.05, "depth": 4,
              "l2_leaf_reg": 3, "subsample": 0.7, "verbose": 100,
              "random_seed": 42, "thread_count": thread_count_default}

lgb_params = {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6,
              "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 1.0,
              "random_state": 42, "n_jobs": thread_count_default, "verbose": 50}

# ========== SHAP 采样相关（控制大小）==========
shap_sample_size = 2000   # 调试时可降到 500

# ========== 绘图函数（不再在函数内 fit 模型，只做 SHAP+绘图）=========
def plot_shap_for_model(model_name, model, X_shap_input, X_all, outdir, clip_val=1500, shap_sample_size=2000):
    """
    model must be already fitted before calling this function.
    X_shap_input: data used as background (e.g. X_train) for possible Explainer fallback
    X_all: full features DataFrame (we will sample from it for shap plotting)
    """
    print(f"\n--- {model_name}: Start SHAP (sampling {shap_sample_size}) — {get_sys_info()} ---", flush=True)
    n_samples = min(shap_sample_size, len(X_all))
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(X_all), size=n_samples, replace=False)
    X_shap = X_all.iloc[idx_sample]

    t_shap0 = time.time()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        # for some models shap_values might be list (multiclass), take first array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e:
        print(f"{model_name}: TreeExplainer failed with {e}; trying shap.Explainer fallback", flush=True)
        # use a small background sample for speed
        bg = X_shap_input.sample(n=min(200, len(X_shap_input)), random_state=42)
        explainer = shap.Explainer(model, bg)
        ev = explainer(X_shap)
        shap_values = getattr(ev, "values", np.array(ev))

    shap_values = np.array(shap_values)
    t_shap1 = time.time()
    print(f"{model_name}: SHAP computed in {t_shap1 - t_shap0:.1f}s — {get_sys_info()}", flush=True)

    # plotting: adapt figure height to number of features to increase line spacing
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)
    sorted_features = [X_all.columns[i] for i in sorted_idx]

    # dynamic figure height: base 9, plus 0.25 per feature (tweakable)
    height = max(9, 0.25 * len(sorted_features))
    width = 8
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["mathtext.fontset"] = "stix"

    feature_colors_map = {}
    for feat in X_all.columns:
        if feat in static_vars:
            feature_colors_map[feat] = static_colors
        elif feat in dynamic_vars:
            feature_colors_map[feat] = dynamic_colors
        else:
            feature_colors_map[feat] = "#B0B0B0"

    shap_values_clipped = np.clip(shap_values, -clip_val, clip_val)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1,1,1)

    for pos, feat in enumerate(sorted_features):
        vals = shap_values_clipped[:, X_all.columns.get_loc(feat)]
        # 若样本仍然很多则再子采样绘图点，防止绘图非常慢
        if len(vals) > 2000:
            idx_plot = np.random.choice(len(vals), size=2000, replace=False)
            vals_plot = vals[idx_plot]
            y = np.random.normal(pos, 0.1, size=len(vals_plot))
        else:
            vals_plot = vals
            y = np.random.normal(pos, 0.1, size=len(vals_plot))
        ax.scatter(vals_plot, y, s=10, c=[feature_colors_map[feat]], alpha=0.7)

    ytick_labels = [pretty_name(f) for f in sorted_features]
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("SHAP value", fontsize=14, fontweight="bold")
    ax.set_title(f"Global SHAP Feature Importance ({model_name})", fontsize=16, fontweight="bold")

    static_patch  = mpatches.Patch(color=static_colors,  label="Static")
    dynamic_patch = mpatches.Patch(color=dynamic_colors, label="Dynamic")
    ax.legend(handles=[static_patch, dynamic_patch], loc="best", fontsize=12)

    # increase left margin to avoid label截断 and add some top/bottom padding
    plt.subplots_adjust(left=0.35, right=0.96, top=0.96, bottom=0.04, hspace=0.3)

    fname = os.path.join(outdir, f"figure5global_shap_static_dynamic_{model_name}.png")
    plt.savefig(fname, dpi=600)
    plt.close()
    print(f"{model_name}: Saved {fname}  (plot height={height:.1f})", flush=True)

# ========== RF incremental训练（分批构建树）=========
def train_rf_with_progress(X_train, y_train, X_val=None, y_val=None,
                           n_estimators=200, batch_size=20, rf_kwargs=None):
    if rf_kwargs is None:
        rf_kwargs = {}
    print("\n--- RF incremental training start ---", flush=True)
    rf = RandomForestRegressor(warm_start=True, n_jobs=1, **{k:v for k,v in rf_kwargs.items() if k != 'n_jobs'})
    built = 0
    rf.n_estimators = 0
    start_total = time.time()
    while built < n_estimators:
        to_build = min(batch_size, n_estimators - built)
        rf.n_estimators = built + to_build
        t0 = time.time()
        rf.fit(X_train, y_train)   # warm_start=True -> 增量构造树
        t1 = time.time()
        built = rf.n_estimators
        status = f"RF: built {built}/{n_estimators} trees (batch {to_build}) in {t1-t0:.1f}s"
        if X_val is not None and y_val is not None:
            try:
                ypred = rf.predict(X_val)
                rmse = np.sqrt(((ypred - y_val) ** 2).mean())
                status += f", val_RMSE={rmse:.4f}"
            except Exception:
                status += ", val predict failed"
        status += f"  | {get_sys_info()}"
        print(status, flush=True)
    total_time = time.time() - start_total
    rf.warm_start = False
    rf.set_params(n_estimators=n_estimators)  # finalize
    print(f"RF complete: total build time {total_time:.1f}s", flush=True)
    return rf

# ========== 训练并绘图（改为 CatBoost -> RF -> LightGBM）=========
'''
# 1) CatBoost — 放在最前面（并打印迭代信息）
print("\n--- CatBoost: start (will print iterations as it runs) ---", flush=True)
cat_model = CatBoostRegressor(
    iterations=cat_params["iterations"],
    learning_rate=cat_params["learning_rate"],
    depth=cat_params["depth"],
    l2_leaf_reg=cat_params["l2_leaf_reg"],
    subsample=cat_params["subsample"],
    verbose=cat_params["verbose"],
    random_seed=cat_params["random_seed"],
    thread_count=cat_params["thread_count"]
)
t0 = time.time()
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
t1 = time.time()
print(f"CatBoost fitted in {t1 - t0:.1f}s ({get_sys_info()})", flush=True)
plot_shap_for_model("CatBoost", cat_model, X_train, X_all, output_dir, shap_sample_size=shap_sample_size)

# 2) RF incremental训练以便看到进度
rf = train_rf_with_progress(X_train, y_train, X_test, y_test,
                            n_estimators=rf_total_trees, batch_size=rf_batch, rf_kwargs=rf_params)
plot_shap_for_model("RF", rf, X_train, X_all, output_dir, shap_sample_size=shap_sample_size)
'''
# 3) LightGBM
print("\n--- LightGBM: start (may print training messages) ---", flush=True)
lgb_model = lgb.LGBMRegressor(
    n_estimators=lgb_params["n_estimators"],
    learning_rate=lgb_params["learning_rate"],
    max_depth=lgb_params["max_depth"],
    min_child_samples=lgb_params["min_child_samples"],
    subsample=lgb_params["subsample"],
    colsample_bytree=lgb_params["colsample_bytree"],
    random_state=lgb_params["random_state"],
    n_jobs=lgb_params["n_jobs"]
    # note: do NOT pass verbose to fit() in some lightgbm versions
)
t0 = time.time()
# >>> FIX: remove 'verbose' kwarg here because some lightgbm versions throw TypeError
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
t1 = time.time()
print(f"LightGBM fitted in {t1 - t0:.1f}s ({get_sys_info()})", flush=True)
plot_shap_for_model("LGB", lgb_model, X_train, X_all, output_dir, shap_sample_size=shap_sample_size)

print("\nAll done — three SHAP figures saved in folder:", output_dir, flush=True)
