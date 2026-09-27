from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
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