# COLUMNS
avg_temperature = "AvgTemperature"

# METHODS OF CORRELATION
XGB = "XGBOOST"
COR = "CORRELATION"

# ERROR TYPES
MAE = "MAE"
MEDAE = "MEDAE"
MSE = "MSE"
MSLE = "MSLE"
RMSE = "RMSE"

# PREDICTION TYPES
EST_EXOG = "estimated_exog"
REAL_EXOG = "real_exog"
BASIC = "basic"

# PARAM TYPES
INIT = "init"
FIT = "fit"


# Lists
error_metrics = [MAE, MEDAE, MSE]
sklearn_models = ["XGBRegressor","LinearRegression", "Ridge", "ElasticNet", "Lasso", "LassoLars", "BayesianRidge",
                  "TweedieRegressor", "SGDRegressor", "PassiveAggressiveRegressor", "HuberRegressor",
                  "TheilSenRegressor", "SVR", "NuSVR", "GradientBoostingRegressor", "RandomForestRegressor",
                  "HistGradientBoostingRegressor", "DecisionTreeRegressor", "ExtraTreeRegressor", "LGBMRegressor"]

statsmodels_models = ["ExponentialSmoothing","HoltWintersResultsWrapper","SARIMAX","SARIMAXResultsWrapper","ThetaModel","ThetaModelResults","ETSModel","ETSResultsWrapper"]

exogless_models = ["BATS","TBATS","Model"]