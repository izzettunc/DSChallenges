import preprocess
import models
import constants as c

if __name__ == '__main__':
    train_test_split_ratio = 0.75
    forecast_horizon = 360
    result_path = "/mnt/HDD/my-files/git-projects/DSChallenges/amsterdam-temperature-forecast/results"

    train, test, scaled_train, scaled_test, scalers = preprocess.get_data(train_test_split_ratio)
    models.run_models((train, test, scaled_train, scaled_test, scalers), c.avg_temperature, forecast_horizon, result_path)

