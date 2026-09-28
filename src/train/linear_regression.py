from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.config import RANDOM_SEED
from numpy import sqrt


def linear_regression(x_train, y_train, x_test, y_test) -> tuple[LinearRegression, dict]:
    model = LinearRegression()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return {
        "name" : "linear regression",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        }
    }


def optimize_linear_regression(x_train, y_train, x_test, y_test):

    model = LinearRegression()

    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
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
        "name" : "linear regression",
        "model" : "linear_regression",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        },
        "best_params" : grid_search.best_params_
    }