import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import joblib
import random
import os
import features
import logging

# data1 = print(int(os.path.getsize("D://data_judgement/out_features/real_data.csv")) / 1024 / 1024)

logging.basicConfig(filename="D://data_judgement/remote/xgb_bagging.log", filemode="w",
                    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data = features.data
data.drop(labels=["城市", "站点名称", "站点编号", "pm10mark"], axis=1, inplace=True)
data = data.replace([-np.inf, np.inf], np.nan).dropna(axis=0)

# 测试集
time1 = data[data["时间"] >= "2022-08-15 23:00:00"]
test_f = data[data["时间"].isin(list(time1["时间"]))]

data.drop(labels=["时间"], axis=1, inplace=True)
test_f.drop(labels=["时间"], axis=1, inplace=True)
data = data.drop(test_f.index, axis=0)

x = data["pm25mark"]
y = data.drop(labels=["pm25mark"], axis=1)

num_model = 5
num_solution = int(num_model / 2) + 1

xgb_params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
              "max_depth": [10, 20, 30, 40, 50],
              "n_estimators": [100, 300, 500, 700, 900],
              "subsample": [1, 3, 5, 7, 9]}

# 三个xgboost
def bagging_train():
    for i in range(num_model):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        xgb = XGBClassifier(tree_method='gpu_hist')   # gpu加速
        # estimator = GridSearchCV(estimator=xgb, param_grid=xgb_params, n_jobs=-1, cv=5)
        estimator = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=20, n_jobs=-1)
        estimator.fit(x_train, y_train)
        print("第{}轮最好的参数:{}".format(i+1, estimator.best_params_))
        print("第{}轮最好的模型:{}".format(i+1, estimator.best_estimator_))
        logging.info(f"第{i+1}轮最好的参数:{estimator.best_params_}")
        logging.info(f"第{i+1}轮最好的模型:{estimator.best_estimator_}")

        with open("D://data_judgement/out_features/xgb_bagging_{}.pkl".format(i+1), "wb") as xgb_estimator:
            joblib.dump(estimator, xgb_estimator)

        y_pred = estimator.predict(x_test)
        print("第{}轮:真实值中异常值的数量:{}".format(i+1, len(y_test[y_test == 1])))
        print("第{}轮:预测集中异常值的数量:{}".format(i+1, len(y_pred[y_pred == 1])))
        print("第{}轮：异常值预测准确的数量：{}，真实值中预测准确的比例：{}".format(i+1,
                                                         len(y_test[(y_test == y_pred) & (y_test == 1)]),
                                                         len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(
                                                         y_test[y_test == 1])))
        logging.info(f"训练时测试集的准确率为:")
        logging.info(f"第{i+1}轮:真实值中异常值的数量:{len(y_test[y_test == 1])}")
        logging.info(f"第{i+1}轮:预测集中异常值的数量:{len(y_pred[y_pred == 1])}")
        logging.info(f"第{i+1}轮：异常值预测准确的数量：{len(y_test[(y_test == y_pred) & (y_test == 1)])}，"
                     f"真实值中预测准确的比例：{len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_test[y_test == 1])}")

# bagging
def bagging_test(test_f):
    y_test_pred = pd.DataFrame(test_f["pm25mark"], index=range(len(test_f)), columns=["pm25mark"])
    y_test_pred["xgb_pred"] = 0
    test_f = test_f.drop(labels=["pm25mark"], axis=1)
    for j in range(num_model):
        with open("D://data_judgement/out_features/xgb_bagging_{}.pkl".format(j+1), "rb") as xgb_model:
            xgb_model = joblib.load(xgb_model)
        y_pred = xgb_model.pred(test_f)
        y_test_pred["xgb_pred_{}".format(j+1)] = y_pred
        y_test_pred["xgb_pred_{}".format(j + 1)] = y_test_pred["xgb_pred_{}".format(j+1)].astype(int)

    for k in range(len(y_test_pred)):
        n = 0
        for m in range(num_model):
            n += y_test_pred["xgb_pred_{}".format(m+1)][k]
        if n >= num_solution:
            y_test_pred["xgb_pred"][k] = 1
        else:
            y_test_pred["xgb_pred"][k] = 0

    print("真实值中异常值的数量:{}".format(len(y_test_pred[y_test_pred["pm25mark"] == 1])))
    print("预测集中异常值的数量:{}".format(len(y_test_pred[y_test_pred["xgb_pred"] == 1])))
    print("异常值预测准确的数量：{}，真实值中预测准确的比例：{}".format(len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]),
                                                                len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]) / len(
                                                                y_test_pred[y_test_pred["pm25mark"] == 1])))
    logging.info(f"训练时测试集的准确率为:")
    logging.info(f"真实值中异常值的数量:{len(y_test_pred[y_test_pred['pm25mark'] == 1])}")
    logging.info(f"预测集中异常值的数量:{len(y_test_pred[y_test_pred['xgb_pred'] == 1])}")
    logging.info(f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['pm25mark'] == y_test_pred['xgb_pred']) & (y_test_pred['pm25mark'] == 1)])}，"
                 f"真实值中预测准确的比例：{len(y_test_pred[(y_test_pred['pm25mark'] == y_test_pred['xgb_pred']) & (y_test_pred['pm25mark'] == 1)]) / len(y_test_pred[y_test_pred['pm25mark'] == 1])}")



