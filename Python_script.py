import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─── Config ───────────────────────────────────────────────────────────
API_KEY    = "......"
MIN_E_GPa  = 80.0    
MIN_BG_eV  = 3.0     
MAX_EHULL  = 0.05   
IONIC_IONS = {"Li", "Na", "K"}
TOP_N      = 5

RADIOACTIVE = {
    "Tc","Pm","Po","At","Rn","Fr","Ra","Ac",
    "Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
    "Es","Fm","Md","No","Lr"
}

# Physical upper bounds for elastic moduli.
# Diamond (hardest known material): E ≈ 1220 GPa, K ≈ 440 GPa.
# Anything above these caps is a failed/unconverged DFT calculation.
K_MAX = 700    
G_MAX = 500    
E_MAX = 1300   

# ======================================================================
# 1.  FETCH DATA
#
#  The summary endpoint lists k_vrh/g_vrh as fields but returns all-NaN.
#  Elastic moduli MUST come from the dedicated elasticity endpoint.
# ======================================================================
print("=" * 60)
print("STEP 1 — Fetching data from Materials Project")
print("=" * 60)

SUMMARY_FIELDS = [
    "material_id", "formula_pretty",
    "energy_above_hull", "formation_energy_per_atom",
    "band_gap", "is_metal",
    "density", "nsites", "volume",
]

with MPRester(API_KEY, use_document_model=False) as mpr:
    summary_docs = list(tqdm(
        mpr.materials.summary.search(
            fields=SUMMARY_FIELDS, num_chunks=None, chunk_size=1000
        ), desc="Summary"
    ))

df_summary = pd.DataFrame(summary_docs)
df_summary["material_id"] = df_summary["material_id"].astype(str).str.strip()
print(f"  Summary:      {len(df_summary):,} materials")

# bulk_modulus / shear_modulus return as nested dicts {"vrh":..., ...}
elastic_rows = []
with MPRester(API_KEY, use_document_model=False) as mpr:
    elastic_docs = list(tqdm(
        mpr.materials.elasticity.search(
            fields=["material_id", "bulk_modulus", "shear_modulus"],
            num_chunks=None, chunk_size=1000
        ), desc="Elasticity"
    ))

for doc in elastic_docs:
    bulk  = doc.get("bulk_modulus",  {})
    shear = doc.get("shear_modulus", {})
    elastic_rows.append({
        "material_id": str(doc.get("material_id", "")).strip(),
        "k_vrh": bulk.get("vrh")  if isinstance(bulk,  dict) else None,
        "g_vrh": shear.get("vrh") if isinstance(shear, dict) else None,
    })

df_elastic = pd.DataFrame(elastic_rows)
df_elastic["material_id"] = df_elastic["material_id"].astype(str).str.strip()
print(f"  Elasticity:   {len(df_elastic):,} materials")

df = df_summary.merge(df_elastic, on="material_id", how="inner")
print(f"  After merge:  {len(df):,} materials with complete data")

for col in ["k_vrh","g_vrh","band_gap","energy_above_hull",
            "formation_energy_per_atom","density","nsites","volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["k_vrh","g_vrh","band_gap"]).copy()

# ── Linking Formula 1: Young's modulus  E = 9KG / (3K + G) ──────────
df["youngs_modulus"] = (9 * df["k_vrh"] * df["g_vrh"]) / (3 * df["k_vrh"] + df["g_vrh"])

# ══════════════════════════════════════════════════════════════════════
# FIX 1 — Remove outliers from the elastic dataset.
#
# The MP elasticity table contains some failed / unconverged DFT runs
# that produce physically impossible values (e.g. k_vrh = 8×10¹² GPa).
# These corrupt the XGBoost training and produce MAE ~ 10⁹ GPa (R²≈0).
# We drop anything outside the physical range of known real materials.
# ══════════════════════════════════════════════════════════════════════
pre = len(df)
df = df[
    (df["k_vrh"]          > 0) & (df["k_vrh"]          < K_MAX) &
    (df["g_vrh"]          > 0) & (df["g_vrh"]          < G_MAX) &
    (df["youngs_modulus"] > 0) & (df["youngs_modulus"] < E_MAX)
].reset_index(drop=True)

print(f"\n  Removed {pre - len(df)} outlier rows (bad DFT elastic runs)")
print(f"  Clean dataset:  {len(df):,} materials")
print(f"  Young's mod:    {df['youngs_modulus'].min():.0f} – {df['youngs_modulus'].max():.0f} GPa")
print(f"  Band gap:       {df['band_gap'].min():.2f} – {df['band_gap'].max():.2f} eV")


# ======================================================================
# 2.  FEATURIZATION — MATMINER MAGPIE
# ======================================================================
print("\n" + "=" * 60)
print("STEP 2 — Featurization with matminer (Magpie)")
print("=" * 60)

df = StrToComposition().featurize_dataframe(
    df, col_id="formula_pretty", ignore_errors=True
)
ep = ElementProperty.from_preset("magpie")
df = ep.featurize_dataframe(df, col_id="composition", ignore_errors=True)

FEAT_COLS = ep.feature_labels()
df = df.dropna(subset=FEAT_COLS).reset_index(drop=True)

print(f"  Materials after featurization : {len(df):,}")
print(f"  Magpie features generated     : {len(FEAT_COLS)}")


# ======================================================================
# 3.  TRAIN SURROGATE MODELS (XGBoost)
# ======================================================================
print("\n" + "=" * 60)
print("STEP 3 — Training XGBoost surrogate models")
print("=" * 60)

X = df[FEAT_COLS].values

XGB = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.8,
           random_state=42, n_jobs=-1, verbosity=0)

def train_surrogate(X, y, label, unit):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(**XGB)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print(f"  {label:32s}  MAE = {mean_absolute_error(y_te, preds):7.2f} {unit}"
          f"   R² = {r2_score(y_te, preds):.3f}")
    return model

model_E  = train_surrogate(X, df["youngs_modulus"].values, "Young's modulus", "GPa")
model_bg = train_surrogate(X, df["band_gap"].values,       "Band gap",        "eV ")


# ======================================================================
# 4.  INVERSE SCREENING — predict properties for every candidate
# ======================================================================
print("\n" + "=" * 60)
print("STEP 4 — Inverse screening")
print("=" * 60)

df["ml_E"]  = model_E.predict(X)
df["ml_bg"] = model_bg.predict(X)

# ── Linking Formula 2: Arrhenius conductivity proxy ───────────────────
KB = 8.617e-5   # eV / K
df["sigma_proxy"] = np.exp(-df["ml_bg"] / (2 * KB * 300))

print(f"  ML Young's mod > {MIN_E_GPa} GPa : {(df['ml_E'] > MIN_E_GPa).sum():,}")
print(f"  ML Band gap    > {MIN_BG_eV} eV  : {(df['ml_bg'] > MIN_BG_eV).sum():,}")


# ======================================================================
# 5.  FILTERS
# ======================================================================
print("\n" + "=" * 60)
print("STEP 5 — Applying target + synthesizability filters")
print("=" * 60)

def has_any(formula, eset):
    try:
        return any(str(el) in eset for el in Composition(formula).elements)
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════════════
# FIX 2 — Filter by actual DFT values, not ML predictions.
#
# We have DFT ground truth for every material in our screening pool.
# Using it directly is more reliable than ML predictions.
# The surrogate (ml_E / ml_bg) remains useful for:
#   (a) materials without DFT elastic data (new compositions)
#   (b) report validation — compare ml vs DFT to prove model quality
# ══════════════════════════════════════════════════════════════════════
df["ok_E"]      = df["youngs_modulus"] >= MIN_E_GPa     # actual DFT Young's mod
df["ok_bg"]     = df["band_gap"]       >= MIN_BG_eV     # actual DFT band gap
df["ok_stable"] = df["energy_above_hull"] <= MAX_EHULL
df["ok_ionic"]  = df["formula_pretty"].apply(lambda f: has_any(f, IONIC_IONS))
df["ok_safe"]   = ~df["formula_pretty"].apply(lambda f: has_any(f, RADIOACTIVE))
df["passes"]    = df[["ok_E","ok_bg","ok_stable","ok_ionic","ok_safe"]].all(axis=1)

for flag, label in [
    ("ok_E",      f"Young's mod (DFT) ≥ {MIN_E_GPa} GPa"),
    ("ok_bg",     f"Band gap (DFT) ≥ {MIN_BG_eV} eV  → σ < 10⁻⁸ S/m"),
    ("ok_stable", f"Ehull ≤ {MAX_EHULL} eV/atom (stable)"),
    ("ok_ionic",  "Contains Li / Na / K  (ionic transport)"),
    ("ok_safe",   "No radioactive elements"),
    ("passes",    "ALL FILTERS"),
]:
    print(f"  {label:48s}  {df[flag].sum():5,}")


# ======================================================================
# 6.  RANK & SELECT TOP 5
# ======================================================================
print("\n" + "=" * 60)
print("STEP 6 — Top 5 candidates")
print("=" * 60)

df["score"] = (
    0.40 * (df["youngs_modulus"].clip(0, 300) / 300) +
    0.30 * (df["band_gap"].clip(0, 8)         / 8)   +
    0.30 * (1 - df["energy_above_hull"] / MAX_EHULL).clip(0, 1)
)

top5 = (
    df[df["passes"]]
    .sort_values("score", ascending=False)
    .head(TOP_N)
    .copy()
)

# Show both DFT ground truth and ML predictions side-by-side
COLS = [
    "material_id", "formula_pretty",
    "energy_above_hull",
    "youngs_modulus",   # DFT — used for filtering
    "ml_E",             # ML prediction — for report validation
    "band_gap",         # DFT
    "ml_bg",            # ML prediction
    "score",
]
COLS = [c for c in COLS if c in top5.columns]

print("\n" + "═" * 80)
print("  TOP 5  —  SOLID-STATE ELECTROLYTE / CATHODE COATING CANDIDATES")
print("  youngs_modulus = DFT ground truth  |  ml_E = surrogate prediction")
print("═" * 80)
print(top5[COLS].to_string(index=False))
print("═" * 80)

top5.to_csv("top5_candidates.csv", index=False)
df.to_csv("all_screened.csv",      index=False)
print("\nSaved:  top5_candidates.csv  |  all_screened.csv")