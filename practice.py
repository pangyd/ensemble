import numpy as np

# x1 = np.ones((10, 1))
# x2 = np.arange(1, 11).reshape(10, 1)
# x = np.hstack((x1, x2))
# y = np.random.randint(2, 15, (10, 1))
# alpha = 0.04

def gradient_function(x, y, theta):
    diff = np.dot(x, theta)
    return 1 / 10 * (np.dot(x, diff - y))

def gradient_decent(x, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(x, y, theta)
    while gradient <= 1e-5:
        theta = theta - alpha * gradient
        gradient = gradient_function(x, y, theta)
    return theta



from ensemble_predata import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import pickle

def search_bagging():
    xgb_params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                  "max_depth": [10, 20, 30, 40, 50],
                  "n_estimators": [100, 300, 500, 700, 900],
                  "subsample": [0.1, 0.3, 0.5, 0.7, 0.9]}

    xgb = XGBClassifier()

    estmator = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=2)

    estimator = BaggingClassifier(base_estimator=estmator, n_estimators=3)

    print("开始训练")
    estimator.fit(x, y)

    with open("bagging.dat", "wb") as model:
        pickle.dump(estimator, model)
    print("训练结束")


from sympy import *
def equation():
    x = Symbol("x")
    print(solve([(x - (x / 2 + 2)) - ((x - (x / 2 + 2)) / 2 - 2) - 5], [x]))
