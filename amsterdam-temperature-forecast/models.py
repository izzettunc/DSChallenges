import traceback

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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from tbats import BATS,TBATS

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
        param_init = parameters[c.INIT] if c.INIT in parameters.keys() else {}
        param_fit = parameters[c.FIT] if c.FIT in parameters.keys() else {}
        if estimator.__name__ in c.sklearn_models:
            self.exog_estimator = ExogPredictor(estimator, parameters).fit(preprocess.get_date_based_variables(self.s_x), self.s_x)
            self.estimator = estimator(**param_init).fit(self.s_x, self.y, **param_fit)
        elif estimator.__name__ in c.exogless_models:
            self.estimator = estimator(**param_init).fit(self.y, **param_fit)
        elif estimator.__name__ in c.statsmodels_models:
            self.estimator = estimator(self.y, **param_init).fit(**param_fit)

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
        elif type(self.estimator).__name__ in c.exogless_models:
            self.pred_y = pd.Series(self.estimator.forecast(len(self.test_y)))
            self.pred_y.index = self.test_y.index
        elif type(self.estimator).__name__ in c.statsmodels_models:
            self.pred_y = self.estimator.forecast(len(self.test_y))

        return self

    def forecast(self, steps):
        start_day = self.test_y.index[len(self.test_y)-1] + DateOffset(days=1)
        forecast_template = pd.DataFrame([], index=pd.date_range(start_day,periods=steps,freq="D"))
        if type(self.estimator).__name__ in c.sklearn_models:
            forecasted_x = self.exog_estimator.predict(preprocess.get_date_based_variables(forecast_template))

            self.forecast_y = pd.DataFrame(self.estimator.predict(forecasted_x))
            self.forecast_y.index = forecast_template.index
            self.forecast_y.columns = [self.target]
        elif type(self.estimator).__name__ in c.exogless_models:
            self.forecast_y = pd.Series(self.estimator.forecast(len(self.test_y)+len(forecast_template)))
            self.forecast_y = self.forecast_y.iloc[len(self.test_y):]
            self.forecast_y.index = forecast_template.index
        elif type(self.estimator).__name__ in c.statsmodels_models:
            self.forecast_y = self.estimator.forecast(len(self.test_y)+len(forecast_template))
            self.forecast_y = self.forecast_y.iloc[len(self.test_y):]

        return self

    def evaluate(self):
        errors = {
            c.EST_EXOG : {},
            c.REAL_EXOG : {},
            c.BASIC : {}
        }
        for metric in c.error_metrics:
            if hasattr(self,"pred_y_est_exog"):
                errors[c.EST_EXOG][metric] = eval(f"{metric}(self.test_y,self.pred_y_est_exog[self.target])")
            if hasattr(self,"pred_y_real_exog"):
                errors[c.REAL_EXOG][metric] = eval(f"{metric}(self.test_y,self.pred_y_real_exog[self.target])")
            if hasattr(self,"pred_y"):
                errors[c.BASIC][metric] = eval(f"{metric}(self.test_y,self.pred_y)")
        if hasattr(self, "pred_y_est_exog"):
            errors[c.EST_EXOG]["RMSE"] = MSE(self.test_y, self.pred_y_est_exog[self.target], squared=False)
        if hasattr(self, "pred_y_real_exog"):
            errors[c.REAL_EXOG]["RMSE"] = MSE(self.test_y, self.pred_y_real_exog[self.target], squared=False)
        if hasattr(self, "pred_y"):
            errors[c.BASIC]["RMSE"] = MSE(self.test_y, self.pred_y, squared=False)

        self.errors = errors

        return self

    def save_report(self, path, error=True, plot=True, forecast=True,train=True):

        if plot:
            if hasattr(self, "pred_y_est_exog"):
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

            if hasattr(self, "pred_y_real_exog"):
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

            if hasattr(self, "pred_y"):
                f_3 = plt.figure(figsize=(20, 10))
                if train:
                    sns.lineplot(data=self.y, legend='brief', label="Train")
                sns.lineplot(data=self.test_y, legend='brief', label="Test")
                sns.lineplot(data=self.pred_y, legend='brief', label="Prediction").set_title(type(self.estimator).__name__)
                if forecast:
                    sns.lineplot(data=self.forecast_y, legend='brief', label="Forecast")
                plt.savefig(f"{path}/{type(self.estimator).__name__}-basic.png")

                f_3.clear()
                plt.close(f_3)



        if error:
            if hasattr(self, "pred_y_est_exog"):
                pd.DataFrame(self.errors[c.EST_EXOG],index=[0]).to_csv(f"{path}/{type(self.estimator).__name__}-est_exog_error.csv", index=False)
            if hasattr(self, "pred_y_real_exog"):
                pd.DataFrame(self.errors[c.REAL_EXOG],index=[0]).to_csv(f"{path}/{type(self.estimator).__name__}-real_exog_error.csv", index=False)
            if hasattr(self, "pred_y"):
                pd.DataFrame(self.errors[c.BASIC],index=[0]).to_csv(f"{path}/{type(self.estimator).__name__}-basic_error.csv", index=False)




def run_models(data_tuple, target, forecast_horizon, report_path):
    train, test, scaled_train, scaled_test, scalers = data_tuple
    predictor = Predictor(train, test, scaled_train, scaled_test, scalers, target)

    sklearn_models = [LinearRegression, Ridge, ElasticNet, Lasso, LassoLars,BayesianRidge,TweedieRegressor,SGDRegressor,
                      PassiveAggressiveRegressor,HuberRegressor,TheilSenRegressor,SVR,NuSVR,GradientBoostingRegressor,
                      RandomForestRegressor,HistGradientBoostingRegressor,DecisionTreeRegressor, ExtraTreeRegressor,
                      LGBMRegressor,XGBRegressor]

    statsmodels_models = [ETSModel, ThetaModel, ExponentialSmoothing, SARIMAX]

    # TODO: Fix the issue of both bats models returns Model class as result. Therefore they write on each other.
    exogless_models = [TBATS, BATS]
    models = exogless_models+statsmodels_models+sklearn_models

    model_params = {
        TweedieRegressor.__name__: {
            c.INIT : {"max_iter": 1000}
        },
        SARIMAX.__name__:{
            c.FIT : {"disp":0},
            c.INIT: {"order":(2,1,2),"seasonal_order":(2,1,2,12)} #Test params
        },
        ETSModel.__name__:{
            c.FIT : {"disp":0}
        }
    }

    for model in models:
        try:
            print(f"{model.__name__} is calculating...")
            params = model_params[model.__name__] if model.__name__ in model_params.keys() else None
            predictor.fit(model, params).predict().evaluate().forecast(forecast_horizon).save_report(report_path)
            print("Done")
        except Exception as e:
            print(e)
            traceback.print_exc()

