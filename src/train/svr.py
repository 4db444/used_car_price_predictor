from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from src.config import RANDOM_SEED
import numpy as np

def svr(x_train, y_train, x_test, y_test):

    model = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1)),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "name" : "svr",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        }
    }


def optimize_svr(x_train, y_train, x_test, y_test):

    model = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR()),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    param_grid = {
        "regressor__svr__kernel": ["rbf"],
        "regressor__svr__C": [10, 100, 1000],
        "regressor__svr__gamma": ["scale", 0.01, 0.1],
        "regressor__svr__epsilon": [0.01, 0.1, 0.5]
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
    rmse = np.sqrt(mse)

    return {
        "name" : "svr",
        "model" : "svr",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        },
        "best_params" : grid_search.best_params_
    }