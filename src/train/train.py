from pandas import DataFrame
from src import save_data
from ..config import UNOPTIMIZED_MODELS_STATS
from .linear_regression import linear_regression
from .svr import svr
from .xgboost import xgboost
from .random_forest import random_forest


def train(x_train, y_train, x_test, y_test):
    linear_regression_result = linear_regression(x_train, y_train, x_test, y_test)
    svr_result = svr(x_train, y_train, x_test, y_test)
    xgboost_result = xgboost(x_train, y_train, x_test, y_test)
    random_forest_result = random_forest(x_train, y_train, x_test, y_test)

    report = {
        linear_regression_result["name"] : linear_regression_result["stats"],
        svr_result["name"] : svr_result["stats"],
        xgboost_result["name"] : xgboost_result["stats"],
        random_forest_result["name"] : random_forest_result["stats"],
    }

    df = DataFrame(
        report,
        index=list(svr_result["stats"].keys()),
    )

    save_data(df, UNOPTIMIZED_MODELS_STATS, "models_report", True)
