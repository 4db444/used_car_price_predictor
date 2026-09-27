from sklearn.metrics import mean_squared_error, r2_score
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