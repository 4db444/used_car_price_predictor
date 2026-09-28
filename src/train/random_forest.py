from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from src.config import RANDOM_SEED
from numpy import sqrt

def random_forest (x_train, y_train, x_test, y_test) : 

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=10,
        min_samples_split=10,
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return {
        "name" : "random forest",
        "model" : "random_forest",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        }
    }


def optimize_random_forest (x_train, y_train, x_test, y_test):

    model = RandomForestRegressor(
        random_state=RANDOM_SEED
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 1.0]
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
        "name" : "random forest",
        "model" : "random_forest",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        },
        "best_params" : grid_search.best_params_
    }