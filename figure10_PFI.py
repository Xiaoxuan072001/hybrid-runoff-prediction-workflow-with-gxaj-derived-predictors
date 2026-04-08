'''
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_CSV = "LSTM_normalized.csv"
OUTPUT_DIR = "PFI_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'lambda': 0.5,
    'alpha': 0.5,
    'seed': RANDOM_STATE
}
NUM_BOOST_ROUND = 300

# PFI settings
N_REPEATS = 30
METRIC_NAME = "NSE"
TOPN_PLOT = 10

# colors
COLOR_SHAP = "#ee7564"   # SHAP color
COLOR_PFI  = "#6cc6d8"   # PFI color
COLOR_ERR  = "gray"      # errorbar color (PFI std)

# figure styling tuned for A4 landscape (clear font sizes)
FIGSIZE = (11, 5)      # width x height in inches (fits A4 landscape)
TITLE_FSIZE = 18
LABEL_FSIZE = 14
TICK_FSIZE = 12
# Make SHAP and PFI annotations identical in size and weight
ANNOT_FSIZE = 12        # used for both SHAP and PFI now
LEGEND_FSIZE = 12
BAR_HEIGHT = 0.32

# ---------------- Matplotlib global styling ----------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ---------------- Load data ----------------
print("Loading data:", DATA_CSV)
df = pd.read_csv(DATA_CSV)
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

# feature selection (same rules as your original script)
static_vars = ["DEM_ave", "DEM_range", "Slope_ave", "TWI_ave", "LU(cropland)", "LU(forest)", "LU(grass)", "Area", "SoilC", "SoilD", "LLM", "LUM", "CG", "LU(snow)"]
dynamic_vars = ["Q_XAJ", "P", "Pt-1", "NDVI", "Wind", "SM1", "SM2", "SM3", "SM4", "LST", "t", "tmin", "tmax", "Td", "ET", "Rn", "RH", "Solar", "Thermal", "h", "AI", "Ep", "Sproxy"]
exclude_vars = [ "Kes", "C", "CS", "Kech", "Kei", "Keg", "Xech", "Xes", "Xei", "Xeg", "K", "CI", "ISA"]

reserved_cols = ["Q_obs", "date", "ID"] + exclude_vars
features = [c for c in df.columns if c not in reserved_cols]

if "Q_obs" not in df.columns:
    raise ValueError("Input CSV must contain 'Q_obs' column.")

X_all = df[features].copy()
y_all = df["Q_obs"].copy()

print(f"Detected {len(features)} features. Example: {features[:10]}")

# ---------------- Split & train ----------------
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train/Test sizes: {len(X_train)}/{len(X_test)}")

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("Training XGBoost...")
t0 = time.time()
model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
t1 = time.time()
print(f"Training finished in {t1 - t0:.1f}s")

# -------------- metrics ------------------
def nse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1 - np.sum((y_true - y_pred)**2) / denom

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2))

def pbias(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return 100.0 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-12)

def kge(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    r = np.corrcoef(y_true, y_pred)[0,1]
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    gamma = (np.std(y_pred) / (np.mean(y_pred)+1e-12)) / (np.std(y_true) / (np.mean(y_true)+1e-12))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

metric_dict = {"NSE": nse, "RMSE": rmse, "KGE": kge, "PBIAS": pbias}
if METRIC_NAME not in metric_dict:
    raise ValueError("METRIC_NAME must be one of 'NSE','RMSE','KGE','PBIAS'")
metric_fn = metric_dict[METRIC_NAME]

# baseline
baseline_pred = model.predict(xgb.DMatrix(X_test))
baseline_metrics = {
    "NSE": nse(y_test.values, baseline_pred),
    "RMSE": rmse(y_test.values, baseline_pred),
    "PBIAS": pbias(y_test.values, baseline_pred),
    "KGE": kge(y_test.values, baseline_pred)
}
print("Baseline (test) metrics:", baseline_metrics)

greater_is_better = {"NSE": True, "KGE": True, "RMSE": False, "PBIAS": False}
rng = np.random.RandomState(RANDOM_STATE)

# ---------------- PFI computation -------------
def compute_pfi(model, X, y, metric_fn, n_repeats=30, rng=None, verbose=True):
    X = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)
    baseline_score = metric_fn(y.values, model.predict(xgb.DMatrix(X)))
    if verbose:
        print(f"Baseline {METRIC_NAME}: {baseline_score:.6f}")
    rows = []
    features = X.columns.tolist()
    iterator = features if not verbose else tqdm(features, desc="PFI features")
    for feat in iterator:
        perm_scores = []
        for r in range(n_repeats):
            shuffled = X[feat].sample(frac=1.0, replace=False, random_state=rng.randint(0, 10**9)).values
            X_perm = X.copy()
            X_perm[feat] = shuffled
            pred = model.predict(xgb.DMatrix(X_perm))
            perm_scores.append(metric_fn(y.values, pred))
        perm_scores = np.array(perm_scores)
        if greater_is_better[METRIC_NAME]:
            importance = baseline_score - perm_scores.mean()
        else:
            importance = perm_scores.mean() - baseline_score
        rows.append({
            "feature": feat,
            "pfi_mean": float(importance),
            "pfi_std": float(perm_scores.std(ddof=0)),
            "perm_mean_score": float(perm_scores.mean())
        })
    pfi_df = pd.DataFrame(rows).sort_values("pfi_mean", ascending=False).reset_index(drop=True)
    return pfi_df, baseline_score

print(f"Running PFI (n_repeats={N_REPEATS}, metric={METRIC_NAME}) ...")
t0 = time.time()
pfi_df, baseline_score_for_metric = compute_pfi(model, X_test, y_test, metric_fn, n_repeats=N_REPEATS, rng=rng, verbose=True)
t1 = time.time()
print(f"PFI finished in {t1 - t0:.1f}s")

# ---------------- SHAP computation --------------
print("Computing SHAP values (test set)...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_arr = np.asarray(shap_values) if not isinstance(shap_values, list) else np.asarray(shap_values).sum(axis=0)
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_test.columns, "shap_mean_abs": mean_abs_shap})
    shap_df.sort_values("shap_mean_abs", ascending=False, inplace=True)
    shap_df.reset_index(drop=True, inplace=True)
    print("SHAP done.")
except Exception as e:
    print("SHAP failed:", e)
    shap_df = pd.DataFrame({"feature": X_test.columns, "shap_mean_abs": np.nan})

# ---------------- merge & save ----------------
merge_df = pd.merge(shap_df, pfi_df, on="feature", how="outer")
merge_df["pfi_mean"].fillna(0.0, inplace=True)
merge_df["pfi_std"].fillna(0.0, inplace=True)

valid_idx = (~merge_df["shap_mean_abs"].isna()) & (~merge_df["pfi_mean"].isna())
if valid_idx.sum() >= 3:
    rho, pval = spearmanr(merge_df.loc[valid_idx, "shap_mean_abs"], merge_df.loc[valid_idx, "pfi_mean"])
else:
    rho, pval = np.nan, np.nan

out_xlsx = os.path.join(OUTPUT_DIR, f"PFI_SHAP_results_{METRIC_NAME}.xlsx")
with pd.ExcelWriter(out_xlsx) as writer:
    pfi_df.to_excel(writer, sheet_name="PFI", index=False)
    shap_df.to_excel(writer, sheet_name="SHAP", index=False)
    merge_df.to_excel(writer, sheet_name="Merge", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k,v in baseline_metrics.items()]).to_excel(writer, sheet_name="Baseline", index=False)
print("Saved results to", out_xlsx)

# ---------------- plot Top-10 (styled) --------------
print("Preparing Top-10 plot...")

# Map feature names to display names (with math-style subscripts in upright Times New Roman)
def disp_name(feat):
    if feat == "Sproxy" or feat == "S_proxy":
        return r"$\mathrm{S}_{\mathrm{proxy}}$"
    # Q_XAJ mapped to QG_{XAJ} as requested
    if feat == "Q_XAJ" or feat == "QXAJ" or feat == "QGXAJ":
        return r"$\mathrm{Q}_{\mathrm{GXAJ}}$"
    if feat == "Pt-1" or feat == "Pt_1":
        return r"$\mathrm{P}_{\mathrm{t-1}}$"
    # fallback: escape underscores to show as plain text
    if "_" in feat:
        return feat.replace("_", r"\_")
    return feat

plot_df = merge_df.sort_values("shap_mean_abs", ascending=False).head(TOPN_PLOT).copy().reset_index(drop=True)

# normalize for display
def normalize(s):
    mn = np.nanmin(s)
    mx = np.nanmax(s)
    if np.isclose(mx, mn):
        return np.zeros_like(s, dtype=float)
    return (s - mn) / (mx - mn)

plot_df["shap_norm"] = normalize(plot_df["shap_mean_abs"])
plot_df["pfi_norm"]  = normalize(plot_df["pfi_mean"])

# scale pfi_std into normalized scale (by pfi range)
pmin, pmax = (plot_df["pfi_mean"].min(), plot_df["pfi_mean"].max())
den = (pmax - pmin) if not np.isclose(pmax, pmin) else 1.0
plot_df["pfi_std_norm"] = plot_df["pfi_std"] / den

# y positions: top1 on top
y_pos = np.arange(len(plot_df))[::-1]

# display labels (mathtext where needed)
display_labels = [disp_name(f) for f in plot_df["feature"].tolist()]

fig, ax = plt.subplots(figsize=FIGSIZE)

# bars
ax.barh(y_pos + BAR_HEIGHT/2, plot_df["shap_norm"].values, height=BAR_HEIGHT,
        label="SHAP mean(|SHAP|)", color=COLOR_SHAP, edgecolor="none")
ax.barh(y_pos - BAR_HEIGHT/2, plot_df["pfi_norm"].values, height=BAR_HEIGHT,
        label=f"PFI mean (n={N_REPEATS}) [{METRIC_NAME}]", color=COLOR_PFI, edgecolor="none")

# errorbars (PFI std) in gray
ax.errorbar(plot_df["pfi_norm"].values, y_pos - BAR_HEIGHT/2, xerr=plot_df["pfi_std_norm"].values,
            fmt='none', ecolor=COLOR_ERR, alpha=0.9, capsize=4, linewidth=1.2)

# axis labels + title
ax.set_yticks(y_pos)
ax.set_yticklabels(display_labels, fontsize=TICK_FSIZE, fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Normalized importance (0–1)", fontsize=LABEL_FSIZE, fontweight="bold")
ax.set_title(f"SHAP vs PFI — Top {TOPN_PLOT} (sorted by SHAP)", fontsize=TITLE_FSIZE, fontweight="bold")
ax.xaxis.set_tick_params(labelsize=TICK_FSIZE)
ax.yaxis.set_tick_params(labelsize=TICK_FSIZE)

# set x-axis strictly 0.0 - 1.0 ticks, but visual xlim slightly larger so annotations fit
ax.set_xlim(0.0, 1.05)        # visual limit: 1.05 (keeps ticks 0.0-1.0 but room for labels)
ax.set_xticks(np.linspace(0.0, 1.0, 6))
ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.0, 1.0, 6)], fontsize=TICK_FSIZE)

# legend
leg = ax.legend(frameon=False, fontsize=LEGEND_FSIZE)
for text in leg.get_texts():
    text.set_fontweight("bold")

# ------------------ Modified annotation logic: show both SHAP & PFI (PFI now bold, no parentheses) ------------------
for i, row in plot_df.iterrows():
    ypos_v = y_pos[i]

    # SHAP original value (left, bold) - place slightly inside left of SHAP bar
    x_sh = 0.01
    shap_text = f"{row['shap_mean_abs']:.3f}"
    ax.text(x_sh, ypos_v + BAR_HEIGHT/2, shap_text, va='center', ha='left',
            fontsize=ANNOT_FSIZE, fontweight="bold", color="black", family="Times New Roman")

    # PFI annotation text (now WITHOUT parentheses and bold, same style as SHAP)
    pfi_text = f"{row['pfi_mean']:.3f} ± {row['pfi_std']:.3f}"

    # Preferred external position: a bit after the longer bar
    preferred_x = max(row["shap_norm"], row["pfi_norm"]) + 0.035

    # Visual limit for placing outside labels (we keep ticks 0.0-1.0 but allow labels to 1.03)
    outside_limit = 1.03

    if preferred_x <= outside_limit:
        # place outside, left-aligned, bold, same font size as SHAP
        xpos = preferred_x
        ha = 'left'
        ax.text(xpos, ypos_v - BAR_HEIGHT/2, pfi_text, va='center', ha=ha,
                fontsize=ANNOT_FSIZE, fontweight="bold", color="black", family="Times New Roman")
    else:
        # not enough room outside: place inside PFI bar, right-aligned, bold
        # compute an inside-x that is slightly left of the bar end
        inside_x = row["pfi_norm"] - 0.015
        # ensure it's within [0.02, 0.98]
        inside_x = min(max(inside_x, 0.02), 0.98)
        ax.text(inside_x, ypos_v - BAR_HEIGHT/2, pfi_text, va='center', ha='right',
                fontsize=ANNOT_FSIZE, fontweight="bold", color="black", family="Times New Roman")

# make layout tight but with a bit more right margin to be safe
plt.tight_layout()
plt.subplots_adjust(right=0.96)

# save high-res (suitable for A4)
out_png = os.path.join(OUTPUT_DIR, f"SHAP_PFI_top{TOPN_PLOT}_{METRIC_NAME}_final_v6.png")
out_pdf = os.path.join(OUTPUT_DIR, f"SHAP_PFI_top{TOPN_PLOT}_{METRIC_NAME}_final_v6.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

# save top table
plot_df.to_csv(os.path.join(OUTPUT_DIR, f"SHAP_PFI_top{TOPN_PLOT}_table_final_v6.csv"), index=False)

# -------------- summary print ----------------
print("\n=== Summary ===")
print("Baseline metrics (test):")
for k, v in baseline_metrics.items():
    print(f"  {k}: {v:.4f}")
print(f"Spearman between SHAP(mean|.|) and PFI({METRIC_NAME}): rho={rho:.4f}, p={pval}")
print("Top features (by SHAP):")
print(plot_df[["feature", "shap_mean_abs", "pfi_mean", "pfi_std"]])
print(f"\nOutputs saved under: {os.path.abspath(OUTPUT_DIR)}")
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pfi_xgb_rf_only.py

Train XGBoost and RandomForest, compute PFI (permutation importance) on test set,
plot stacked panels (top: XGBoost PFI top10, bottom: RandomForest PFI top10).
Only PFI is displayed (no SHAP). Save CSV/Excel and PNG/PDF.
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_CSV = "LSTM_normalized.csv"
OUTPUT_DIR = "PFI_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost params
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'lambda': 0.5,
    'alpha': 0.5,
    'seed': RANDOM_STATE
}
NUM_BOOST_ROUND = 300

# PFI settings
N_REPEATS = 30
METRIC_NAME = "NSE"
TOPN = 10

# plot style
PFI_COLOR = "#6cc6d8"
ERR_COLOR = "gray"
FIGSIZE = (10, 4)     # 横向宽一点，适合论文横图
BAR_HEIGHT = 0.50

TITLE_FSIZE = 18
LABEL_FSIZE = 14
TICK_FSIZE = 12
ANNOT_FSIZE = 14
LEGEND_FSIZE = 14

# Matplotlib: Times New Roman, upright mathtext
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
# set 'it' to Times New Roman as upright to avoid italics
plt.rcParams['mathtext.it'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

rng = np.random.RandomState(RANDOM_STATE)

# ---------------- helpers ----------------
def nse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1 - np.sum((y_true - y_pred)**2) / denom

metric_fn = nse

def compute_pfi_xgb(model, X, y, metric_fn, n_repeats=30, rng=None, verbose=True):
    Xloc = X.copy().reset_index(drop=True)
    yloc = y.copy().reset_index(drop=True)
    baseline_pred = model.predict(xgb.DMatrix(Xloc))
    baseline_score = metric_fn(yloc.values, baseline_pred)
    if verbose:
        print(f"Baseline {METRIC_NAME}: {baseline_score:.6f}")
    rows = []
    features = Xloc.columns.tolist()
    iterator = features if not verbose else tqdm(features, desc="PFI features")
    for feat in iterator:
        perm_scores = []
        for r in range(n_repeats):
            seed = rng.randint(0, 2**31 - 1) if rng is not None else None
            shuffled = Xloc[feat].sample(frac=1.0, replace=False, random_state=seed).values
            Xp = Xloc.copy(); Xp[feat] = shuffled
            pred = model.predict(xgb.DMatrix(Xp))
            perm_scores.append(metric_fn(yloc.values, pred))
        perm_scores = np.asarray(perm_scores)
        importance = baseline_score - perm_scores.mean()   # drop in NSE (positive -> important)
        rows.append({
            "feature": feat,
            "pfi_mean": float(importance),
            "pfi_std": float(perm_scores.std(ddof=0)),
            "perm_mean_score": float(perm_scores.mean())
        })
    pfi_df = pd.DataFrame(rows).sort_values("pfi_mean", ascending=False).reset_index(drop=True)
    return pfi_df, baseline_score

def pretty_name(feat):
    if feat in ("Sproxy","S_proxy"):
        return r"$\mathrm{S}_{\mathrm{proxy}}$"
    if feat in ("Q_XAJ","QXAJ","QGXAJ"):
        return r"$\mathrm{Q}_{\mathrm{GXAJ}}$"
    if feat in ("Pt-1","Pt_1"):
        return r"$\mathrm{P}_{\mathrm{t-1}}$"
    if "_" in feat:
        return feat.replace("_", r"\_")
    return feat

# ---------------- load data ----------------
print("Loading:", DATA_CSV)
df = pd.read_csv(DATA_CSV)
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

exclude_vars = ["Kes","C","CS","Kech","Kei","Keg","Xech","Xes","Xei","Xeg","K","CI","ISA"]
reserved_cols = ["Q_obs","date","ID"] + exclude_vars
features = [c for c in df.columns if c not in reserved_cols]

if "Q_obs" not in df.columns:
    raise ValueError("CSV must contain 'Q_obs'")

X_all = df[features].copy()
y_all = df["Q_obs"].copy()
print(f"Found {len(features)} features. Example: {features[:10]}")

# ---------------- split & train ----------------
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train/Test sizes: {len(X_train)}/{len(X_test)}")

print("Training XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
print("XGBoost trained.")

# ---------------- compute PFI ----------------
print(f"Computing PFI (n_repeats={N_REPEATS}) ...")
pfi_df, baseline_score = compute_pfi_xgb(model, X_test, y_test, metric_fn, n_repeats=N_REPEATS, rng=rng, verbose=True)
print("Top 10 PFI:\n", pfi_df.head(10))

# ---------------- prepare Top-N ----------------
top_df = pfi_df.head(TOPN).copy().reset_index(drop=True)
top_df["display"] = top_df["feature"].apply(pretty_name)

# Decide normalization: if max <=1 use raw as bar length; else normalize by max (rare)
pmax = top_df["pfi_mean"].max() if not np.isnan(top_df["pfi_mean"].max()) else 0.0
if pmax <= 1.0:
    top_df["pfi_plot"] = top_df["pfi_mean"].clip(lower=0.0)   # clip negative for plotting
    top_df["pfi_std_plot"] = top_df["pfi_std"]
    normalized_flag = False
else:
    top_df["pfi_plot"] = top_df["pfi_mean"] / pmax
    top_df["pfi_std_plot"] = top_df["pfi_std"] / pmax
    normalized_flag = True

print(f"pfi_max = {pmax:.6f}; normalized_flag = {normalized_flag}")

# ---------------- save numeric ----------------
out_csv = os.path.join(OUTPUT_DIR, f"PFI_XGB_top{TOPN}.csv")
out_xlsx = os.path.join(OUTPUT_DIR, f"PFI_XGB_top{TOPN}.xlsx")
top_df.to_csv(out_csv, index=False)
with pd.ExcelWriter(out_xlsx) as writer:
    top_df.to_excel(writer, sheet_name="TopPFI", index=False)
    pfi_df.to_excel(writer, sheet_name="AllPFI", index=False)
print("Saved numeric outputs:", out_csv, out_xlsx)

# ---------------- plotting ----------------
fig, ax = plt.subplots(figsize=FIGSIZE)

y_pos = np.arange(len(top_df))[::-1]
ax.barh(y_pos, top_df["pfi_plot"].values, height=BAR_HEIGHT, color=PFI_COLOR, edgecolor="none")
ax.errorbar(top_df["pfi_plot"].values, y_pos, xerr=top_df["pfi_std_plot"].values, fmt='none',
            ecolor=ERR_COLOR, capsize=4, linewidth=1.2)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_df["display"].tolist(), fontsize=TICK_FSIZE, fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Relative PFI (0–1)", fontsize=LABEL_FSIZE, fontweight="bold")
ax.set_title(f"XGBoost PFI — Top {TOPN}", fontsize=TITLE_FSIZE, fontweight="bold")
ax.xaxis.set_tick_params(labelsize=TICK_FSIZE)
ax.set_xlim(0.0, 1.05)
ax.set_xticks(np.linspace(0.0, 1.0, 6))
ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.0,1.0,6)], fontsize=TICK_FSIZE)

# annotate mean ± std (raw) beside bars (bold, Times New Roman upright)
for i, row in top_df.iterrows():
    ypos_i = y_pos[i]
    mean_raw = row["pfi_mean"]
    std_raw  = row["pfi_std"]
    xpos = float(row["pfi_plot"]) + 0.03
    if xpos <= 1.03:
        ha = "left"
    else:
        # place inside bar if not enough room
        xpos = float(row["pfi_plot"]) - 0.02
        if xpos < 0.01: xpos = 0.01
        ha = "right"
    ax.text(xpos, ypos_i, f"{mean_raw:.3f} ± {std_raw:.3f}",
            va="center", ha=ha, fontsize=ANNOT_FSIZE, fontweight="bold",
            family="Times New Roman", fontstyle="normal", color="black")

# if normalized, add small note
if normalized_flag:
    ax.text(0.995, 0.02, "PFI normalized by model max for plotting", va="bottom", ha="right",
            fontsize=9, transform=ax.transAxes, color="gray")

plt.tight_layout()
plt.subplots_adjust(right=0.96)

out_png = os.path.join(OUTPUT_DIR, f"PFI_XGB_top{TOPN}_only.png")
out_pdf = os.path.join(OUTPUT_DIR, f"PFI_XGB_top{TOPN}_only.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved figure:", out_png)
print("Saved PDF:", out_pdf)
print("Done. Outputs under:", os.path.abspath(OUTPUT_DIR))
