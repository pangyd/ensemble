from ensemble_predata import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cof import COF
from pyod.models.knn import KNN


def isolate(x_train, y_train):
    y = pd.DataFrame(y_train.values, columns=["y_test"])
    clf = IsolationForest()
    clf.fit(x_train)
    y_pred = clf.predict(x_train)
    y["y_pred"] = y_pred
    y["y_pred"][y["y_pred"] == 1] = 0
    y["y_pred"][y["y_pred"] == -1] = 1
    print(len(y[y["y_test"] == 1]))
    print(len(y[y["y_pred"] == 1]))
    print(len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]))


def lof(x_train, y_train):
    y = pd.DataFrame(y_train.values, columns=["y_test"])
    l = LocalOutlierFactor(n_neighbors=1)
    y["y_pred"] = l.fit_predict(x_train[["pm25", "pm10"]])
    y["y_pred"][y["y_pred"] == 1] = 0
    y["y_pred"][y["y_pred"] == -1] = 1
    print(len(y[y["y_test"] == 1]))
    print(len(y[y["y_pred"] == 1]))
    print(len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]))


def cof(x_train, y_train):
    y = pd.DataFrame(y_train.values, columns=["y_test"])
    c = COF(contamination=0.5, n_neighbors=5)
    y["y_pred"] = c.fit_predict(x_train[["pm25", "pm10"]])
    print("真实值中异常值个数：", len(y[y["y_test"] == 1]))
    print("预测之中异常值个数：", len(y[y["y_pred"] == 1]))
    print("预测正确的个数", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]))
    print("准确率：", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]) / len(y[y["y_test"] == 1]))
    print("召回率：", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]) / len(y[y["y_pred"] == 1]))


def knn(x_train, y_train):
    y = pd.DataFrame(y_train.values, columns=["y_test"])
    k = KNN(contamination=0.5, n_neighbors=3, method="mean")
    # k.fit(x_train)
    y["y_pred"] = k.fit_predict(x_train[["pm25", "pm10"]])
    print("真实值中异常值个数：", len(y[y["y_test"] == 1]))
    print("预测之中异常值个数：", len(y[y["y_pred"] == 1]))
    print("预测正确的个数", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]))
    print("准确率：", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]) / len(y[y["y_test"] == 1]))
    print("召回率：", len(y[(y["y_test"] == y["y_pred"]) & (y["y_pred"] == 1)]) / len(y[y["y_pred"] == 1]))


# y_train = test_f["pm25mark"]
# x_train = test_f.drop(labels=["pm25mark"], axis=1)
# knn(x_train, y_train)
