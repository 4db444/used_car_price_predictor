car-price-prediction/
│
├── README.md                          # Project overview, setup, usage instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore venv, __pycache__, data dumps, etc.
├── LICENSE                            # Optional
│
├── data/
│   ├── raw/                           # Original, untouched dataset
│   │   └── car_data.csv
│   ├── processed/                     # Cleaned/encoded/scaled data ready for training
│   │   ├── train.csv
│   │   └── test.csv
│   └── external/                      # Any additional reference data (optional)
│
├── notebooks/
│   ├── 01_exploration.ipynb           # EDA: stats, histograms, correlations
│   ├── 02_cleaning.ipynb              # Missing values, duplicates, outliers, encoding
│   ├── 03_baseline_models.ipynb       # Step 2: default models + metrics
│   ├── 04_hyperparameter_tuning.ipynb # Step 3: GridSearch/RandomizedSearch
│   ├── 05_model_comparison.ipynb      # Step 4: residuals, scatter plots, final choice
│   └── 06_final_model_export.ipynb    # Step 5: joblib/pickle export
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Paths, constants, random seed, hyperparams
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py               # Load raw CSV
│   │   └── preprocess.py              # Cleaning, encoding, scaling, train/test split
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py          # Feature engineering / encoding logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py             # Training pipeline (all 4 models)
│   │   ├── tune_model.py              # Hyperparameter search
│   │   ├── evaluate_model.py          # RMSE, MAE, R², residual plots
│   │   └── predict_model.py           # Load model, run prediction
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py               # Reusable plotting functions
│
├── models/
│   ├── baseline/                      # Saved baseline models (optional, for comparison)
│   └── final_model.pkl                # Final exported model (joblib/pickle)
│
├── reports/
│   ├── figures/                       # Saved plots (heatmaps, residuals, scatter plots)
│   │   ├── correlation_heatmap.png
│   │   ├── residuals_plot.png
│   │   └── pred_vs_actual.png
│   └── model_comparison.csv           # Summary DataFrame of all models' metrics
│
├── app/
│   ├── app.py                         # Streamlit/Flask interface for predictions
│   ├── templates/                     # If Flask: HTML templates
│   └── static/                        # If Flask: CSS/JS assets
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py             # Unit tests for data cleaning
│   ├── test_train_model.py            # Unit tests for training pipeline
│   └── test_predict.py                # Unit tests for prediction interface
│
└── docs/
    ├── project_plan.md                # Jira Epics/tickets summary or export
    └── architecture.md                # Optional: pipeline/architecture diagram