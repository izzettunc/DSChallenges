# COLUMNS
avg_temperature = "AvgTemperature"

# METHODS
XGB = "XGBOOST"
COR = "CORRELATION"

#Errors
MAE = "MAE"
MEDAE = "MEDAE"
MSE = "MSE"
MSLE = "MSLE"
RMSE = "RMSE"

EST_EXOG = "estimated_exog"
REAL_EXOG = "real_exog"


# Lists
error_metrics = [MAE, MEDAE, MSE]
sklearn_models = ["XGBRegressor","LinearRegression", "Ridge", "ElasticNet", "Lasso", "LassoLars", "BayesianRidge",
                  "TweedieRegressor", "SGDRegressor", "PassiveAggressiveRegressor", "HuberRegressor",
                  "TheilSenRegressor", "SVR", "NuSVR", "GradientBoostingRegressor", "RandomForestRegressor",
                  "HistGradientBoostingRegressor", "DecisionTreeRegressor", "ExtraTreeRegressor", "LGBMRegressor"]
