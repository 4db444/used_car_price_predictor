from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from src.config import RANDOM_SEED
from numpy import sqrt

def random_forest (x_train, y_train, x_test, y_test) : 

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return {
        "name" : "random forest",
        "stats" : {
            "MSE" : mse,
            "RMSE" : rmse,
            "R2" : r2
        }
    }