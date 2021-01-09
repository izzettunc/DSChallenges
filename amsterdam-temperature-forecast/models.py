from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import preprocess

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class ExogPredictor:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y: pd.DataFrame):
        self.models = {}
        for column in y.columns:
            estimator = self.estimator()
            estimator.fit(x, y[column])
            self.models[column] = estimator

        return self

    def predict(self, x):
        predictions = {}
        for feature in self.models.keys():
            prediction = self.models[feature].predict(x)
            predictions[feature] = prediction

        return pd.DataFrame(predictions)


def linear_regression(train,test,scaled_train,scaled_test,scalers,target):
    y = train[target].to_frame()
    x = train.loc[:, train.columns != target]

    s_x = scaled_train.loc[:, train.columns != target]

    test_y = test[target].to_frame()
    test_x = test.loc[:, train.columns != target]

    test_s_x = scaled_test.loc[:, train.columns != target]

    # Training
    lr_exog = ExogPredictor(LinearRegression).fit(preprocess.get_date_based_variables(s_x),s_x)
    lr = LinearRegression().fit(s_x, y)

    # Predictions
    predicted_x = lr_exog.predict(preprocess.get_date_based_variables(test_s_x))
    predicted_y_1 = pd.DataFrame(lr.predict(predicted_x))
    predicted_y_2 = pd.DataFrame(lr.predict(test_s_x))
    predicted_y_1.index = predicted_y_2.index = test_y.index
    predicted_y_1.columns = predicted_y_2.columns = [target]

    error_1 = mean_absolute_error(test_y[target], predicted_y_1[target])
    error_2 = mean_absolute_error(test_y[target], predicted_y_2[target])

    plt.figure()
    sns.lineplot(data=test_y, x=test_y.index, y=target,legend='brief', label="Real")
    sns.lineplot(data=predicted_y_1, x=predicted_y_1.index, y=target, legend='brief', label="Prediction").set_title(error_1)
    plt.show()

    plt.figure()
    sns.lineplot(data=test_y, x=test_y.index, y=target, legend='brief', label="Real")
    sns.lineplot(data=predicted_y_2, x=predicted_y_2.index, y=target, legend='brief', label="Prediction").set_title(error_2)
    plt.show()


