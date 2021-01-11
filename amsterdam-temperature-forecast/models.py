from sklearn.linear_model import (
    LinearRegression, Ridge, ElasticNet, Lasso,
    LassoLars,BayesianRidge,TweedieRegressor,SGDRegressor,
    PassiveAggressiveRegressor,HuberRegressor,TheilSenRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MEDAE, mean_squared_error as MSE, mean_squared_log_error as MSLE


import constants as c

import preprocess

import pandas as pd
from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt
import seaborn as sns

class ExogPredictor:
    def __init__(self, estimator, parameters=None):
        self.estimator = estimator
        self.parameters = {} if parameters is None else parameters

    def fit(self, x, y: pd.DataFrame):
        self.models = {}
        for column in y.columns:
            estimator = self.estimator(**self.parameters)
            estimator.fit(x, y[column])
            self.models[column] = estimator

        return self

    def predict(self, x):
        predictions = {}
        for feature in self.models.keys():
            prediction = self.models[feature].predict(x)
            predictions[feature] = prediction

        return pd.DataFrame(predictions)


class Predictor:
    def __init__(self,train,test,scaled_train,scaled_test,scalers,target):
        self.x = train.loc[:, train.columns != target]
        self.y = train[target]
        self.s_x = scaled_train.loc[:, train.columns != target]
        self.test_x = test.loc[:, train.columns != target]
        self.test_y = test[target]
        self.test_s_x = scaled_test.loc[:, train.columns != target]
        self.scalers = scalers
        self.target = target

    def fit(self, estimator, parameters=None):
        parameters = {} if parameters is None else parameters
        if estimator.__name__ in c.sklearn_models:
            self.exog_estimator = ExogPredictor(estimator, parameters).fit(preprocess.get_date_based_variables(self.s_x), self.s_x)
            self.estimator = estimator(**parameters).fit(self.s_x, self.y)

        return self

    def predict(self):
        if type(self.estimator).__name__ in c.sklearn_models:
            predicted_x = self.exog_estimator.predict(preprocess.get_date_based_variables(self.test_s_x))

            # Prediction with estimated exog
            self.pred_y_est_exog = pd.DataFrame(self.estimator.predict(predicted_x))

            # Prediction with real values of exog
            self.pred_y_real_exog = pd.DataFrame(self.estimator.predict(self.test_s_x))

            self.pred_y_est_exog.index = self.pred_y_real_exog.index = self.test_y.index
            self.pred_y_est_exog.columns = self.pred_y_real_exog.columns = [self.target]

        return self

    def forecast(self, steps):
        start_day = self.test_y.index[len(self.test_y)-1] + DateOffset(days=1)
        forecast_template = pd.DataFrame([], index=pd.date_range(start_day,periods=steps,freq="D"))
        if type(self.estimator).__name__ in c.sklearn_models:
            forecasted_x = self.exog_estimator.predict(preprocess.get_date_based_variables(forecast_template))

            self.forecast_y = pd.DataFrame(self.estimator.predict(forecasted_x))
            self.forecast_y.index = forecast_template.index
            self.forecast_y.columns = [self.target]

        return self

    def evaluate(self):
        errors = {
            c.EST_EXOG : {},
            c.REAL_EXOG : {}
        }
        for metric in c.error_metrics:
            errors[c.EST_EXOG][metric] = eval(f"{metric}(self.test_y,self.pred_y_est_exog[self.target])")
            errors[c.REAL_EXOG][metric] = eval(f"{metric}(self.test_y,self.pred_y_real_exog[self.target])")
        errors[c.EST_EXOG]["RMSE"] = MSE(self.test_y, self.pred_y_est_exog[self.target], squared=False)
        errors[c.REAL_EXOG]["RMSE"] = MSE(self.test_y, self.pred_y_real_exog[self.target], squared=False)

        self.errors = errors

        return self

    def save_report(self, path, error=True, plot=True, forecast=True,train=True):

        if plot:
            f_1 = plt.figure(figsize=(20, 10))
            if train:
                sns.lineplot(data=self.y, legend='brief', label="Train")
            sns.lineplot(data=self.test_y, legend='brief', label="Test")
            sns.lineplot(data=self.pred_y_est_exog, x=self.pred_y_est_exog.index, y=self.target, legend='brief', label="Prediction").set_title(type(self.estimator).__name__)
            if forecast:
                sns.lineplot(data=self.forecast_y, x=self.forecast_y.index, y=self.target, legend='brief', label="Forecast")

            plt.savefig(f"{path}/{type(self.estimator).__name__}-est_exog.png")

            f_1.clear()
            plt.close(f_1)

            f_2 = plt.figure(figsize=(20, 10))
            if train:
                sns.lineplot(data=self.y, legend='brief', label="Train")
            sns.lineplot(data=self.test_y, legend='brief', label="Test")
            sns.lineplot(data=self.pred_y_real_exog, x=self.pred_y_real_exog.index, y=self.target, legend='brief', label="Prediction").set_title(type(self.estimator).__name__)
            if forecast:
                sns.lineplot(data=self.forecast_y, x=self.forecast_y.index, y=self.target, legend='brief', label="Forecast")
            plt.savefig(f"{path}/{type(self.estimator).__name__}-real_exog.png")

            f_2.clear()
            plt.close(f_2)



        if error:
           pd.DataFrame(self.errors[c.EST_EXOG],index=[0]).to_csv(f"{path}/{type(self.estimator).__name__}-est_exog_error.csv", index=False)
           pd.DataFrame(self.errors[c.REAL_EXOG],index=[0]).to_csv(f"{path}/{type(self.estimator).__name__}-real_exog_error.csv", index=False)


def run_models(data_tuple, target, forecast_horizon, report_path):
    train, test, scaled_train, scaled_test, scalers = data_tuple
    predictor = Predictor(train, test, scaled_train, scaled_test, scalers, target)

    sklearn_models = [LinearRegression, Ridge, ElasticNet, Lasso, LassoLars,BayesianRidge,TweedieRegressor,SGDRegressor,
                      PassiveAggressiveRegressor,HuberRegressor,TheilSenRegressor,SVR,NuSVR,GradientBoostingRegressor,
                      RandomForestRegressor,HistGradientBoostingRegressor,DecisionTreeRegressor, ExtraTreeRegressor,
                      LGBMRegressor,XGBRegressor]

    sklearn_model_params = {
        TweedieRegressor.__name__: {"max_iter": 1000}
    }

    for model in sklearn_models:
        try:
            print(f"{model.__name__} is calculating...")
            params = sklearn_model_params[model.__name__] if model.__name__ in sklearn_model_params.keys() else None
            predictor.fit(model, params).predict().evaluate().forecast(forecast_horizon).save_report(report_path)
            print("Done")
        except Exception as e:
            print(e)

