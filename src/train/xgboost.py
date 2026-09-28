from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from numpy import sqrt
from src.config import RANDOM_SEED

def xgboost (x_train, y_train, x_test, y_test) :     
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return {
        "name" : "xgboost",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        }
    }


def optimize_xgboost(x_train, y_train, x_test, y_test):

    model = XGBRegressor(
        random_state=RANDOM_SEED
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    print("Best params:")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    y_pred = grid_search.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return {
        "name" : "xgboost",
        "model" : "xgboost",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        },
        "best_params" : grid_search.best_params_
    }