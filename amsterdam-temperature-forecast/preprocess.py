import constants as c

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from adtk.detector import SeasonalAD, AutoregressionAD, InterQuartileRangeAD

import xgboost as xgb

from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler

from datetime import datetime

from functools import reduce
import operator
import math
import pickle

def detect_anomalies(iqr,autoreg,seasonal,data,target):
    anomalies = []
    iqr_detector = InterQuartileRangeAD(**iqr)
    try:
        anomalies_iqr = iqr_detector.fit_detect(data[target])
        anomalies.append(anomalies_iqr)
    except Exception as e:
        print(target)
        print(e)
    autoreg_detector = AutoregressionAD(**autoreg)
    try:
        anomalies_autoreg = autoreg_detector.fit_detect(data[target])
        anomalies_autoreg = anomalies_autoreg.apply(lambda x: True if x == 1 else False )
        anomalies.append(anomalies_autoreg)
    except Exception as e:
        print(target)
        print(e)
    seasonal_detector = SeasonalAD(**seasonal)
    try:
        anomalies_seasonal = seasonal_detector.fit_detect(data[target])
        anomalies.append(anomalies_seasonal)
    except Exception as e:
        print(target)
        print(e)
    anomalies = reduce(operator.or_, anomalies)
    return anomalies


def get_date_based_variables(df):
    data = pd.DataFrame([],index=df.index)
    data["date"] = data.index
    data["dayOfWeek"] = data["date"].dt.dayofweek
    data["quarter"] = data["date"].dt.quarter
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data["dayOfYear"] = data["date"].dt.dayofyear
    data["day"] = data["date"].dt.day
    data["week"] = data["date"].dt.isocalendar().week.astype(int)
    data["recentnessDay"] = (datetime.now() - data.index).days
    data["recentnessMonth"] = (datetime.now() - data.index).days // 30
    data["leapYear"] = data.index.is_leap_year.astype(int)
    data["winter"] = (data["month"]%12 < 3).astype(int)
    data["spring"] = ((data["month"]%12 >= 3) & (data["month"]%12 < 6)).astype(int)
    data["summer"] = ((data["month"]%12 >= 6) & (data["month"]%12 < 9)).astype(int)
    data["autumn"] = ((data["month"]%12 >= 9) & (data["month"]%12 < 12)).astype(int)

    data.drop("date",inplace=True,axis=1)
    return data


def scale_features(data,target,scalers=None):
    if scalers is None:
        scalers = {}
        scaled_data = data.copy(deep=True)
        for feature in scaled_data.columns:
            if feature != target:
                ss = StandardScaler()
                ss.fit(scaled_data[feature].to_frame())

                scalers[feature] = ss

                scaled_data[feature] = ss.transform(scaled_data[feature].to_frame()).reshape(-1)
        return  scaled_data,scalers
    else:
        scaled_data = data.copy(deep=True)
        for feature in scaled_data.columns:
            if feature != target:
                scaled_data[feature] = scalers[feature].transform(scaled_data[feature].to_frame()).reshape(-1)
        return scaled_data


def feature_selection(data, target, method=c.XGB, verbose=False):
    if method == c.COR:
        correlation = data.corr()
        if verbose:
            sns.heatmap(correlation, cmap='Blues', annot=True)
            plt.show()

        return correlation.loc[(correlation[target] > 0.2) & (correlation[target] < 0.8)].index.tolist()
    else:
        xgb_params = {
            'eta': 0.05,
            'max_depth': 10,
            'subsample': 1.0,
            'colsample_bytree': 0.7,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }

        df = data.copy(deep=True)
        y = df[target]
        del df[target]
        x = df

        dtrain = xgb.DMatrix(x, y, feature_names=df.columns.values)
        model = xgb.train(xgb_params, dtrain, num_boost_round=1000)
        importance = model.get_score(importance_type='total_cover')
        imp = pd.DataFrame(importance, index=range(1)).T
        imp.columns = ["Importance"]
        imp = imp.sort_values(by=["Importance"], ascending=False)
        imp /= imp.sum()

        if verbose:
            sns.heatmap(imp, cmap='Blues', annot=True)
            plt.show()

        imp["x"] = range(len(imp))

        # Online is a good parameter but you might wanna get rid of it if it gives a bad accuracy
        kneedle = KneeLocator(imp.x, imp.Importance, curve="convex", direction="decreasing", online=True)
        if verbose:
            kneedle.plot_knee()

        return imp.iloc[0:kneedle.knee].index.tolist()+[target]


def get_data(train_rate):
    major_city_temperatures = pd.read_csv("/mnt/HDD/my-files/workplace/Python/Jupyter Lab/Datas/city_temperature.csv")
    # Amsterdam is only city in the data with "Europe" region and "The Netherlands" country
    amsterdam = major_city_temperatures.loc[(major_city_temperatures.Region == 'Europe') & (major_city_temperatures.Country == 'The Netherlands'), ["Month","Day","Year","AvgTemperature"]]
    # Creating date time indices
    amsterdam["Date"] = amsterdam.apply(lambda x: datetime(day=int(x.Day), month=int(x.Month), year=int(x.Year)), axis=1)
    amsterdam = amsterdam.drop_duplicates(subset=['Date'])
    amsterdam = amsterdam.set_index(["Date"])
    # Setting the frequency
    amsterdam = amsterdam.loc[:, "AvgTemperature"].to_frame()
    amsterdam = amsterdam.asfreq("D")

    # Handling missing value issues
    # According to https://academic.udayton.edu/kissock/http/Weather/source.htm
    # "The data fields in each file posted on this site are: month, day, year, average daily temperature (F). We use "-99" as a no-data flag when data are not available."
    amsterdam.loc[amsterdam.AvgTemperature == -99] = np.nan
    amsterdam = amsterdam.interpolate(method="spline", order=3, limit_direction="both")

    # Train-Test split
    train, test = amsterdam.iloc[:int(len(amsterdam) * train_rate)], amsterdam.iloc[int(len(amsterdam) * train_rate):]

    # Handling anomalies and outliers
    anomalies = detect_anomalies({"c": 1.5}, {"n_steps": 3, "step_size": 79, "c": 3.0, "side": "both"},
                                 {"c": 3.0, "side": "both"}, train, "AvgTemperature")
    train.loc[anomalies] = np.nan
    train = train.interpolate(method="spline", order=3, limit_direction="both")

    # Exogenous variables
    exogenous_feature_for_netherlands = pd.read_csv("/mnt/HDD/my-files/workplace/Python/Jupyter Lab/Datas/weather_data.csv")
    weather_stations_for_netherlands = pd.read_csv("/mnt/HDD/my-files/workplace/Python/Jupyter Lab/Datas/stations.csv",sep=";")

    # Column Descriptions
    # ------------------------
    # - DDVEC Vector mean wind direction in degrees (360 = north, 90 = east, 180 = south, 270 = west, 0 = no wind / variable).
    # - FHVEC Vector mean wind speed (in 0.1 m / s).
    # - FG 24-hour average wind speed (in 0.1 m / s)
    # - FHX Highest hourly mean wind speed (in 0.1 m / s)
    # - FHXH Hour period in which FHX is measured
    # - FHN Lowest hourly mean wind speed (in 0.1 m / s)
    # - FHNH Hour period in which FHN is measured
    # - FXX Highest wind gust (in 0.1 m / s)
    # - FXXH Hour segment in which FXX is measured
    # - TG 24-hour average temperature (in 0.1 degrees Celsius)
    # - TN Minimum temperature (in 0.1 degrees Celsius)
    # - TNH Time slot in which TN is measured
    # - TX Maximum temperature (in 0.1 degrees Celsius)
    # - TXH Time slot in which TX is measured
    # - T10N Minimum temperature at 10 cm height (in 0.1 degrees Celsius)
    # - T10NH 6-hour time slot in which T10N was measured
    # - SQ Duration of sunshine (in 0.1 hours) calculated from the global radiation (-1 for <0.05 hours)
    # - SP Percentage of the longest possible sunshine duration
    # - Q Global radiation (in J / cm2)
    # - DR Duration of precipitation (in 0.1 hours)
    # - RH Daily sum of the precipitation (in 0.1 mm) (-1 for <0.05 mm)
    # - RHX Highest hourly sum of the precipitation (in 0.1 mm) (-1 for <0.05 mm)
    # - RHXH Time slot in which RHX is measured
    # - PG 24-hour average air pressure converted to sea level (in 0.1 hPa) calculated from 24 hour values
    # - PX Highest hourly value of the air pressure converted to sea level (in 0.1 hPa)
    # - PXH Hour segment in which PX is measured
    # - PN Lowest hourly value of the air pressure converted to sea level (in 0.1 hPa)
    # - PNH Hour period in which PN is measured
    # - VVN Minimum visibility occurred
    # - VVNH Time slot in which VVN is measured
    # - VVX Maximum visibility occurred
    # - VVXH Time slot in which VVX is measured
    # - NG 24-hour average cloud cover (degree of cover of the upper air in eighths, 9 = upper air invisible)
    # - UX Maximum relative humidity (in percent)
    # - UXH Time slot in which UX is measured
    # - UN Minimum relative humidity (in percent)
    # - UNH Hour period in which UN is measured
    # - EV24 Reference Crop Evaporation (Makkink) (in 0.1 mm)

    # I selected variables which are the most conducive to forecast in my opinion
    exogenous_feature_for_netherlands = exogenous_feature_for_netherlands.loc[:,["STN","YYYYMMDD","DDVEC","   FG","   SQ","    Q","   RH","   PG","   NG"," EV24 "]]

    # 240 is the code of the closest station to Amsterdam which is in SCHIPHOL
    amsterdam_exog_features = exogenous_feature_for_netherlands.loc[exogenous_feature_for_netherlands.STN == 240]
    amsterdam_exog_features.columns = ["stationNo","date","windDir","windSpeed","sunshineDur","globalRadiation","precipitation","airpressure","cloudCoverage", "evapotranspiration"]

    amsterdam_exog_features.index = pd.to_datetime(amsterdam_exog_features['date'].astype(str), format='%Y%m%d')
    amsterdam_exog_features = amsterdam_exog_features.drop(["date", "stationNo"], axis=1)

    # Removing white space entries in cloud coverage and transforming it to integer
    amsterdam_exog_features.cloudCoverage = amsterdam_exog_features.cloudCoverage.apply(lambda x: x.replace(" ", ""))
    amsterdam_exog_features.loc[amsterdam_exog_features["cloudCoverage"] == "", ["cloudCoverage"]] = np.nan
    amsterdam_exog_features.cloudCoverage = amsterdam_exog_features.cloudCoverage.astype(float)
    amsterdam_exog_features.cloudCoverage = amsterdam_exog_features.cloudCoverage.interpolate(method="linear", limit_direction="both")

    # -1 is used for precipitation measurements which are less than 0.05 mm, changing it to 0 will make much more sense for our algorithms
    amsterdam_exog_features.loc[amsterdam_exog_features["precipitation"] == -1, ["precipitation"]] = 0

    amsterdam_exog_features = amsterdam_exog_features.asfreq("D")

    # Train-Test split
    amsterdam_exog_features_test = amsterdam_exog_features.loc[test.index[0]:]
    amsterdam_exog_features = amsterdam_exog_features.loc[:train.index[len(train) - 1]]

    # Anomaly detection for exog variables
    anomalies = {}
    anomalies["windDir"] = detect_anomalies({"c":1.5},{"n_steps":10,"step_size":9,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"windDir")
    anomalies["precipitation"] = detect_anomalies({"c":1.5},{"n_steps":12,"step_size":7,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"precipitation")
    anomalies["windSpeed"] = detect_anomalies({"c":1.5},{"n_steps":6,"step_size":14,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"windSpeed")
    anomalies["airpressure"] = detect_anomalies({"c":1.5},{"n_steps":9,"step_size":10,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"airpressure")
    anomalies["sunshineDur"] = detect_anomalies({"c":1.5},{"n_steps":3,"step_size":75,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"sunshineDur")
    anomalies["cloudCoverage"] = detect_anomalies({"c":1.5},{"n_steps":6,"step_size":14,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"cloudCoverage")
    anomalies["globalRadiation"] = detect_anomalies({"c":1.5},{"n_steps":3,"step_size":75,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"globalRadiation")
    anomalies["evapotranspiration"] = detect_anomalies({"c":1.5},{"n_steps":3,"step_size":75,"c":3.0,"side":"both"},{"c":3.0,"side":"both"},amsterdam_exog_features,"evapotranspiration")

    # Interpolation of missing data in exog variables
    for column in amsterdam_exog_features.columns:
        amsterdam_exog_features.loc[anomalies[column],column] = np.nan
        amsterdam_exog_features.loc[:,[column]] = amsterdam_exog_features.loc[:,[column]].interpolate(method="linear", limit_direction="both")

    amsterdam_exog_features = pd.concat([amsterdam_exog_features,get_date_based_variables(amsterdam_exog_features)],axis=1)
    amsterdam_exog_features_test = pd.concat([amsterdam_exog_features_test,get_date_based_variables(amsterdam_exog_features_test)],axis=1)

    train = pd.concat([train, amsterdam_exog_features], axis=1)

    test = pd.concat([test, amsterdam_exog_features_test], axis=1)

    # Feature scaling
    scaled_train,scalers = scale_features(train,"AvgTemperature")
    scaled_test = scale_features(test,"AvgTemperature",scalers)

    # Feature selection
    features = feature_selection(train, c.avg_temperature,c.XGB)
    train = train.loc[:, features]
    test = test.loc[:, features]
    scaled_train = scaled_train.loc[:, features]
    scaled_test = scaled_test.loc[:, features]

    return train, test, scaled_train, scaled_test, scalers
