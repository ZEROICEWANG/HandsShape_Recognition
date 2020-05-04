import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


# 回归方程求取函数
def fit(x, y):
    if len(x) != len(y):
        return
    numerator = 0.0
    denominator = 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += np.square((x[i] - x_mean))
    print('numerator:', numerator, 'denominator:', denominator)
    b0 = numerator / denominator
    b1 = y_mean - b0 * x_mean
    return b0, b1


# 定义预测函数
def predit(x, b0, b1):
    return b0 * x + b1


# 求取回归方程
def get_regress(x, y):
    b0, b1 = fit(x, y)
    print('Line is:y = %fx + %f' % (b0, b1))
    pre = np.array([b0 * x[i] + b1 for i in range(len(x))])
    r2 = r2_score(y, pre)
    return b0, b1, pre, r2
