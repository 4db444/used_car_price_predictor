# Used Car Price Predictor

End-to-end **supervised regression** project that predicts a used car's resale price from its
characteristics, and serves the trained model through an interactive **Streamlit** dashboard.

Covers the full lifecycle: EDA → cleaning → feature engineering → training 4 model families →
hyper-parameter tuning → comparison & selection → artifact export → web app.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Viz | seaborn, matplotlib |
| Persistence | joblib |
| App | Streamlit |
| Config / Packaging | python-dotenv, setuptools (`pyproject.toml`) |
| Deployment | Docker, Docker Compose |
| Quality | mypy (+ pandas-stubs, types-seaborn) |
| Workflow | Jupyter notebooks |

---

## Dataset

`data/raw/data.csv` — 4,340 used-car listings (1992–2020). **3,649 rows** after cleaning.

| Column | Type | Role |
|---|---|---|
| `name` | object | Model name (e.g. `Maruti Wagon R LXI Minor`) |
| `year` | float | Manufacturing year |
| `km_driven` | float | Distance covered |
| `fuel` | object | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | object | Dealer / Individual / Trustmark Dealer |
| `transmission` | object | Manual / Automatic |
| `owner` | object | First Owner … Fourth & Above, Test Drive Car |
| `selling_price` | int | **Target** |

**Key EDA findings** (`reports/tables/pre-cleaning/`):
- `selling_price` is strongly right-skewed (mean ≈ 504k, median ≈ 350k, max ≈ 8.9M).
- **Age is the dominant signal**: `year` ↔ price `+0.41`, `year` ↔ `km_driven` `-0.42`.
- `km_driven` ↔ price only `-0.19` — mileage matters, but far less than age.
- Missing values: `km_driven` 3.99%, `fuel`/`owner` 1.98%, `year` 0.99%, `seller_type` 1.50%.

---

## Process

Logic lives in small pure functions under `src/`; the two notebooks only orchestrate.

```
data.csv → load → EDA → clean → features/encode → split/scale → train → tune → evaluate → app
```

**1. EDA** — `full_analyse()` → descriptive stats, missing-value frequency, correlation matrix,
plus histograms / scatter plots / heatmap generated **before and after** cleaning for comparison.

**2. Cleaning** (`clear_dataframe`) — order matters:
1. Drop duplicate listings.
2. Group-aware `ffill`/`bfill` — `fuel` & `transmission` by `name`, `owner` by `year` (a given
   model is far more likely manual/petrol than the global mode).
3. Categorical gaps → column **mode**; numeric gaps → **median of the `year` cohort**, then global median.
4. **Winsorize** `km_driven`, `year`, `selling_price` to Tukey fences (`Q1−1.5·IQR` / `Q3+1.5·IQR`) —
   clamping beats deletion, so expensive cars keep a price signal.
5. Drop duplicates again (winsorization can create identical rows).

**3. Feature engineering** — `age = current_year − year`; `brand = name.split(" ")[0]`
(≈30 categories instead of hundreds); one-hot encode `fuel`, `owner`, `transmission`, `brand`,
`seller_type` → 46 columns. Both builders are idempotent.

**4. Split & scale** — drop `name`/`year` and the target; seeded 80/20 hold-out
(`random_state=RANDOM_SEED`); `StandardScaler` fit on train only.

**5. Train** — `LinearRegression`, `SVR` (standardized + `log1p`/`expm1` target via
`TransformedTargetRegressor`), `XGBoost`, `RandomForest`. Reported as MSE / RMSE / R².

**6. Tune** — `GridSearchCV`, `cv=5`, `scoring="neg_root_mean_squared_error"`, `n_jobs=-1`,
seeded. Best params consolidated in `BEST_PARAMS` (`src/train/evaluate.py`).

**7. Evaluate & select** — adds **MAE** and **Residual_Std**, writes per-model
*predictions-vs-actual* and *residual* plots, then selects:
> take the best R² → keep all models within **0.005** of it → among those pick the
> **lowest residual standard deviation** (most consistent errors).

Exports `models/final_model.joblib` = `{model, best_model_name, feature_columns}`.

**8. App** — sidebar form (brand/fuel/transmission/owner/seller/year/km) → same feature pipeline
as training → `reindex(columns=feature_columns, fill_value=0)` to guarantee an identical 46-column
vector, eliminating train/serve skew → `st.metric` result.

---

## Results

Hold-out set, 20% (`reports/tables/models/post-optimization/models_evaluation_report.csv`):

| Model | RMSE | MAE | R² | Residual Std |
|---|---|---|---|---|
| Linear Regression | 170,263 | 126,773 | 0.682 | 170,267 |
| SVR (tuned) | 156,428 | 106,972 | 0.732 | 154,099 |
| **XGBoost (tuned)** | **148,263** | **104,441** | **0.759** | **148,261** |
| Random Forest (tuned) | 148,979 | 104,866 | 0.757 | 149,052 |

- **Final model: XGBoost** — best R² and lowest residual spread.
- The linear baseline already reaches R² 0.682, confirming price is largely linear depreciation;
  tree ensembles add the remaining ~8 points.
- Random Forest is statistically tied (ΔR² = 0.002); boosting wins on sample efficiency.
- **MAE ≪ RMSE** → errors are right-skewed: most predictions land within ~100k while a few luxury
  cars dominate squared error. Use the model as a **pricing range / sanity check**, not an exact valuation.

---

## Folder Structure

```
used_car_price_predictor/
├── pyproject.toml                # dependencies + package definition
├── docker-compose.yml            # containerized Streamlit app
├── Dockerfile                    # python:3.12-slim, serves on :8501
├── .env                          # RANDOM_SEED (git-ignored)
│
├── data/
│   ├── raw/data.csv              # original dataset
│   └── processed/                # data_cleaned.csv, data_encoded.csv
│
├── notebooks/
│   ├── exploration.ipynb         # phases 1-4
│   └── training.ipynb            # phases 5-7
│
├── src/
│   ├── config.py                 # all paths, column lists, group keys, seed
│   ├── data/                     # laod_data, explore, preprocess, save_data
│   ├── feature_engineering/      # create_features (age, brand), encode_features (one-hot)
│   ├── train/                    # train, optimize, evaluate + 4 model modules
│   ├── models/                   # model artifact package
│   └── visualization/            # histogram, scatter_plot, heatmap helpers
│
├── models/final_model.joblib     # {model, best_model_name, feature_columns}
│
├── reports/
│   ├── tables/                   # pre-cleaning stats + model reports
│   └── figures/                  # pre/post-cleaning plots, model residual plots
│
└── app/streamlit_app.py          # prediction dashboard
```

---

## Setup

```bash
git clone https://github.com/4db444/used_car_price_predictor.git
cd used_car_price_predictor

python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env && echo "RANDOM_SEED=47" >> .env
```

> `RANDOM_SEED` is **required** — `src/config.py:29` reads it with `environ[...]` and will raise
> `KeyError` otherwise. It seeds the split, the CV and the tree models for full reproducibility.

### Run the pipeline

```bash
jupyter lab
```

`exploration.ipynb` → `load_data` → `full_analyse` → `clear_dataframe` → `save_data` →
`create_features` → `encode_features` → `save_data`.

`training.ipynb` → load encoded data → `train_test_split` → `train()` → `optimize()` → `evaluate()`.

### Run the app

```bash
streamlit run app/streamlit_app.py      # http://localhost:8501
```

```bash
docker compose up --build              # same, containerized
```

`docker-compose.yml` bind-mounts `./models`, so a locally retrained `final_model.joblib` is picked
up without a rebuild.

