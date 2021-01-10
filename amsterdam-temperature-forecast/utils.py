import time
from functools import wraps
import constants as c

# def console_info_decorator(function):
#     @wraps(function)
#     def print_info_to_console(*args, **kwargs):
#         s = time.time()
#         print(f"{r'%-15s' % type(result.estimator).__name__} model is running")
#         result = function(*args, **kwargs)
#         print(f"EST EXOG RMSE: {result.errors[c.EST_EXOG][c.RMSE]}\t\tREAL EXOG RMSE: {result.errors[c.EST_EXOG][c.RMSE]}")
#         print(f" Elapsed Time: {r'%.2f' % (time.time() - s)} second")
#         return result