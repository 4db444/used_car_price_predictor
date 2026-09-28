from pandas import DataFrame
from src import save_data
from ..config import OPTIMIZED_MODELS_STATS
from .linear_regression import optimize_linear_regression
from .svr import optimize_svr
from .xgboost import optimize_xgboost
from .random_forest import optimize_random_forest


def optimize(x_train, y_train, x_test, y_test):
    linear_regression_result = optimize_linear_regression(x_train, y_train, x_test, y_test)
    svr_result = optimize_svr(x_train, y_train, x_test, y_test)
    xgboost_result = optimize_xgboost(x_train, y_train, x_test, y_test)
    random_forest_result = optimize_random_forest(x_train, y_train, x_test, y_test)

    report = {
        linear_regression_result["name"] : linear_regression_result["stats"],
        svr_result["name"] : svr_result["stats"],
        xgboost_result["name"] : xgboost_result["stats"],
        random_forest_result["name"] : random_forest_result["stats"],
    }

    df = DataFrame(
        report,
        index=list(linear_regression_result["stats"].keys()),
    )

    save_data(df, OPTIMIZED_MODELS_STATS, "models_report", True)

    best_params = {
        result["model"] : result["best_params"]
        for result in [
            linear_regression_result,
            svr_result,
            xgboost_result,
            random_forest_result,
        ]
    }

    print("\nBest params per model:")
    for model, params in best_params.items():
        print(f"{model} : {params}")

    return {
        "report" : df,
        "best_params" : best_params
    }
