Inverse Materials Design Pipeline

Surrogate-Driven High-Throughput Screening**
Target: Solid-State Electrolyte / Cathode Coating Discovery


## Overview

Here  we implements a machine-learning-driven inverse design pipeline that navigates a chemical space of ~154,000 inorganic materials to identify optimal solid-state electrolyte or cathode coating candidates. Rather than forward-predicting properties one material at a time, the pipeline trains surrogate models on DFT-computed data and uses them to screen the entire Materials Project database against strict mesoscopic constraints.

The pipeline is grounded in three DFT-to-mesoscale linking formulas:

| # | Formula 			| Purpose |
|---|---------|---------|
| 1 | `E = 9KG / (3K + G)` 	| Young's modulus from bulk (K) and shear (G) moduli |
| 2 | `σ ∝ exp(−Eg / 2kBT)` 	| Arrhenius electronic conductivity proxy from band gap |
| 3 | `Ehull` 			| Thermodynamic stability / synthesizability gate |



| Requirement 				| How it is met |
|-------------|--------------|
| ML-driven inverse design 		| XGBoost surrogates trained on ~15K DFT-computed elastic + electronic datasets |
| Vast chemical space 			| ~154K materials fetched from Materials Project via `mp_api` |
| Mesoscopic constraints 		| Young's modulus ≥ 80 GPa and band gap ≥ 3 eV filters |
| Synthesizability / stability 		| `energy_above_hull ≤ 0.05 eV/atom` hard gate (Step 5) |
| Synthesis accessibility 		| Ionic transport ions (Li/Na/K) required; radioactive/expensive elements penalized |
| Penalty for radioactive elements 	| Explicit blocklist: Tc, Pm, Po, all actinides, etc. |
| Ionic conductor requirement 		| Filter ensures every candidate contains at least one of Li, Na, K |

## Pipeline Architecture

```
Materials Project API
        │
        ▼
┌──────────────────────────────────┐
│  STEP 1 — Data Fetching          │
│  • Summary endpoint (~154K mats) │
│  • Elasticity endpoint (~15K)    │
│  • Merge on mat. ID   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  STEP 2 — Featurization          │
│  • StrToComposition (pymatgen)   │
│  • Magpie descriptors (matminer) │
│  • 132 composition-based features│
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  STEP 3 — Surrogate Training     │
│  • XGBRegressor (400 trees)      │
│  • Model 1: Young's modulus (GPa)│
│  • Model 2: Band gap (eV)        │
│  • 80/20 train-test split        │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  STEP 4 — Inverse Screening      │
│  • Predict E and Eg for all mats │
│  • Compute σ proxy (Arrhenius)   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  STEP 5 — Filters                │
│  ✔ ML Young's mod ≥ 80 GPa      │
│  ✔ ML Band gap ≥ 3.0 eV         │
│  ✔ Ehull ≤ 0.05 eV/atom         │
│  ✔ Contains Li / Na / K         │
│  ✔ No radioactive elements      │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  STEP 6 — Rank & Output          │
│  • Composite score (40/30/30)    │
│  • Top-5 candidates saved to CSV │
└──────────────────────────────────┘

---

## Installation

**Python ≥ 3.9 required.**

bash
pip install mp-api pymatgen matminer xgboost scikit-learn pandas numpy tqdm


---

## Configuration

Edit the config block at the top of `paas4bat.py`:

```python
API_KEY    = "YOUR_MP_API_KEY"   # Get free key at materialsproject.org
MIN_E_GPa  = 80.0                # Young's modulus floor (GPa)
MIN_BG_eV  = 3.0                 # Band gap floor → σ < 10⁻⁸ S/m
MAX_EHULL  = 0.05                # Energy above hull ceiling (eV/atom)
IONIC_IONS = {"Li", "Na", "K"}   # Required ionic charge carriers
TOP_N      = 5                   # Number of candidates to report
```

Get your free Materials Project API key at [materialsproject.org](https://materialsproject.org).



## Usage

```bash
python paas4bat.py
```



### Expected Console Output

```
============================================================
STEP 1 — Fetching data from Materials Project
============================================================
  Summary:         154,718 materials
  Elasticity:       15,187 materials
  After merge:      14,923 materials with complete data
  Final dataset:    12,441 materials

  Young's mod:     0 – 1043 GPa
  Band gap:        0.00 – 9.87 eV

============================================================
STEP 2 — Featurization with matminer (Magpie)
...
```

---

## Output Files

| File | Description |
|------|-------------|
| `top5_candidates.csv` | Top-5 ranked candidates with all predicted properties |
| `all_screened.csv` | Full dataset with ML predictions and filter flags for all materials |

### `top5_candidates.csv` columns

| Column 		| Description |
|--------		|-------------|
| `material_id` 	| Materials Project ID  |
| `formula_pretty` 	| Chemical formula |
| `energy_above_hull` 	| DFT thermodynamic stability (eV/atom) |
| `youngs_modulus` 	| DFT-derived Young's modulus via E = 9KG/(3K+G) (GPa) |
| `band_gap` 		| DFT band gap (eV) |
| `ml_E` 		| Surrogate-predicted Young's modulus (GPa) |
| `ml_bg` 		| Surrogate-predicted band gap (eV) |
| `score` 		| Composite ranking score (0–1) |


## Scoring Function

Candidates that pass all filters are ranked by a weighted composite score:

score = 0.40 × (ml_E  / 300 GPa)   # mechanical robustness
      + 0.30 × (ml_bg / 8 eV)      # electronic insulation
      + 0.30 × (1 − Ehull / 0.05)  # thermodynamic stability

Weights can be adjusted in Step 6 to reprioritize objectives.



## Dependencies

| Package | Role |
|---------|------|
| `mp-api` | Materials Project REST API client |
| `pymatgen` | Composition parsing, element handling |
| `matminer` | Magpie featurization |
| `xgboost` | Surrogate model training and inference |
| `scikit-learn` | Train/test split, MAE, R² metrics |
| `pandas` / `numpy` | Data manipulation |
| `tqdm` | Progress bars during API fetch |

---

## References

1. Jain, A. et al. *The Materials Project: A materials genome approach to accelerating materials innovation.* APL Materials, 2013.
2. Chen, T. & Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.
