from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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