from os import makedirs
from numpy import sqrt as np_sqrt
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
from xgboost import XGBRegressor

from src import save_data
from src.config import (
    RANDOM_SEED,
    OPTIMIZED_MODELS_STATS,
    MODELS_EVALUATION_FIGURES_PATH,
    FINAL_MODEL_PATH,
)
import numpy as np


BEST_PARAMS = {
    "linear regression": {"fit_intercept": True, "positive": False},
    "svr": {
        "regressor__svr__kernel": "rbf",
        "regressor__svr__C": 10,
        "regressor__svr__gamma": 0.01,
        "regressor__svr__epsilon": 0.1,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
    },
    "random forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "max_features": 1.0,
        "min_samples_leaf": 1,
        "min_samples_split": 10,
    },
}


def _build_models() -> dict:
    return {
        "linear regression": LinearRegression(**BEST_PARAMS["linear regression"]),
        "svr": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(
                    kernel=BEST_PARAMS["svr"]["regressor__svr__kernel"],
                    C=BEST_PARAMS["svr"]["regressor__svr__C"],
                    gamma=BEST_PARAMS["svr"]["regressor__svr__gamma"],
                    epsilon=BEST_PARAMS["svr"]["regressor__svr__epsilon"],
                ),
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "xgboost": XGBRegressor(
            random_state=RANDOM_SEED,
            **BEST_PARAMS["xgboost"],
        ),
        "random forest": RandomForestRegressor(
            random_state=RANDOM_SEED,
            **BEST_PARAMS["random forest"],
        ),
    }


def _save_comparison_plots(x_test, y_test, models: dict) -> None:
    makedirs(MODELS_EVALUATION_FIGURES_PATH, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(x_test)

        fig = pyplot.figure(figsize=(10, 4))

        pyplot.subplot(1, 2, 1)
        pyplot.scatter(y_test, y_pred, alpha=0.6, s=12)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        pyplot.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        pyplot.xlabel("Actual price")
        pyplot.ylabel("Predicted price")
        pyplot.title(f"{name} - Predictions vs Actual")

        pyplot.subplot(1, 2, 2)
        residuals = y_test - y_pred
        pyplot.scatter(y_pred, residuals, alpha=0.6, s=12)
        pyplot.axhline(0, color="r", linestyle="--", linewidth=1)
        pyplot.xlabel("Predicted price")
        pyplot.ylabel("Residuals")
        pyplot.title(f"{name} - Residuals")

        pyplot.tight_layout()
        pyplot.savefig(
            MODELS_EVALUATION_FIGURES_PATH / f"{name.replace(' ', '_')}.png"
        )
        pyplot.close()


def evaluate(x_train, y_train, x_test, y_test) -> dict:
    models = _build_models()

    report = {}
    residual_stds = {}

    for name, model in models.items():
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        residuals = y_test - y_pred

        report[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np_sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Residual_Std": np_sqrt(residuals.var()),
        }
        residual_stds[name] = residuals.std()

    _save_comparison_plots(x_test, y_test, models)

    df = DataFrame(
        report,
        index=["MSE", "RMSE", "MAE", "R2", "Residual_Std"],
    )

    save_data(df, OPTIMIZED_MODELS_STATS, "models_evaluation_report", True)

    best_r2 = max(report.items(), key=lambda item: item[1]["R2"])
    candidates = [
        name
        for name, stats in report.items()
        if stats["R2"] >= best_r2[1]["R2"] - 0.005
    ]
    best_model_name = min(candidates, key=lambda name: residual_stds[name])
    final_model = models[best_model_name]

    dump(
        {
            "model": final_model,
            "best_model_name": best_model_name,
            "feature_columns": list(x_train.columns),
        },
        FINAL_MODEL_PATH,
    )

    print(f"\nFinal model selected: {best_model_name}")
    print(f"R2: {report[best_model_name]['R2']:.4f}")
    print(f"RMSE: {report[best_model_name]['RMSE']:.2f}")
    print(f"MAE: {report[best_model_name]['MAE']:.2f}")
    print(f"Residual_Std: {report[best_model_name]['Residual_Std']:.2f}")

    return {
        "report": df,
        "final_model": final_model,
        "best_model_name": best_model_name,
    }