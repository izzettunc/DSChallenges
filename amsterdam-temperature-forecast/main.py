import preprocess
import models
import constants as c

if __name__ == '__main__':
    train, test, scaled_train, scaled_test, scalers = preprocess.get_data(0.75)
    models.linear_regression(train, test, scaled_train, scaled_test, scalers, c.avg_temperature)

