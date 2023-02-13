import numpy as np
import pandas as pd

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

    x = pd.DataFrame([[2, 4], [2, 5], [3, 6]], columns=["a", "b"])
    x["c"] = x["a"] + x["b"]
    # x.sort_values(by=["a"], axis=0, ascending=False, inplace=True)
    # x = x.value_counts(normalize=True)

    from sklearn.preprocessing import OneHotEncoder, label_binarize
    # ont_hot = OneHotEncoder(sparse=False)
    # y_test = ont_hot.fit_transform(x["a"].values)
    y_test = label_binarize(x["a"].values, classes=[2, 3])


def vt(x):
    from sklearn.feature_selection import VarianceThreshold

    print(x.shape)
    vt = VarianceThreshold(threshold=1)
    x = vt.fit_transform(x)
    print(x.shape)


def one_hot():
    from sklearn.preprocessing import OneHotEncoder
    data = [["a", "b", "a", "c"], ["1", "2", "3", "5"]]
    ohe = OneHotEncoder(categories=[["a", "1"], ["b", "2"], ["a", "3"], ["c", "5"]])
    trans = ohe.fit(data)
    print(trans.get_feature_names())
    trans = trans.transform(data).toarray()
    print(trans)


def double_indicator():
    nums = [4, 2, 4, 0, 0, 3, 0, 5, 1, 0]
    print(nums.count(0))
    i = 0
    j = 0  # 移动的步数
    k = 0
    if len(nums) > 1:
        while i < len(nums):
            if nums[i] == 0 & i == 0:
                nums[i:-1] = nums[i + 1:]
                nums[-1] = 0
            elif nums[i] == 0 & i != 0:
                nums[i:-1] = nums[i + 1:]
                nums[-1] = 0
            elif nums[i] != 0:
                i += 1
            j += 1
            if j == (len(nums)):
                break
    print(nums)


from sklearn.preprocessing import Binarizer, OneHotEncoder


if __name__ == '__main__':
    w = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
    w = pd.DataFrame(w, index=range(len(w)), columns=["a", "b", "c", "d", "e"])
    w["f"] = w.sum(axis=1).apply(lambda x: 1 if x > 2 else 0)
    # x["f"] = x.sum(axis=1)
    print(w)




