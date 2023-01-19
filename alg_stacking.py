import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
import pickle
from ensemble_predata import *
from copy import deepcopy
import itertools
import warnings

warnings.filterwarnings("ignore")


def sklearn_stacking():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 创建基分类器
    xgb = XGBClassifier()
    lg = LGBMClassifier()
    rf = RandomForestClassifier()

    lr = LogisticRegression()

    sclf = StackingClassifier(classifiers=[xgb, lg, rf], meta_classifier=lr)

    sclf.fit(x_train, y_train)

    with open("stacking_model.dat", "wb") as stack_estimator:
        pickle.dump(sclf, stack_estimator)

    y_pred = sclf.predict(x_test)
    # y_pred_meta = sclf.predict_meta_features(x_test)
    y_test_pred = pd.DataFrame([list(y_test), list(y_pred)], columns=["y_test", "y_pred"])

    # print("两种预测相同的比率：", len(y_new[y_new["y_pred"] == y_new["y_pred_meta"]]) / len(y_new))
    print("真实值中异常值的数量:{}".format(len(y_test_pred[y_test_pred["y_test"] == 1])))
    print("预测集中异常值的数量:{}".format(len(y_test_pred[y_test_pred["y_pred"] == 1])))
    print("异常值预测准确的数量：{}，真实值中预测准确的比例：{}".format(len(y_test_pred[(y_test_pred["y_test"] == y_test_pred["y_pred"]) & (y_test_pred["y_test"] == 1)]),
                                                                len(y_test_pred[(y_test_pred["y_test"] == y_test_pred["y_pred"]) & (y_test_pred["y_test"] == 1)]) / len(
                                                                y_test_pred[y_test_pred["y_test"] == 1])))
    print("异常值预测准确的数量：{}，预测集中预测准确的比例：{}".format(len(y_test_pred[(y_test_pred["y_test"] == y_test_pred["y_pred"]) & (y_test_pred["y_test"] == 1)]),
                                                                len(y_test_pred[(y_test_pred["y_test"] == y_test_pred["y_pred"]) & (y_test_pred["y_test"] == 1)]) / len(
                                                                y_test_pred[y_test_pred["y_pred"] == 1])))
    logging.info(f"训练时测试集的准确率为:")
    logging.info(f"真实值中异常值的数量:{len(y_test_pred[y_test_pred['y_test'] == 1])}")
    logging.info(f"预测集中异常值的数量:{len(y_test_pred[y_test_pred['y_pred'] == 1])}")
    logging.info(
        f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)])}，"
        f"真实值中预测准确的比例：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)]) / len(y_test_pred[y_test_pred['y_test'] == 1])}")
    logging.info(
        f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)])}，"
        f"预测集中预测准确的比例：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)]) / len(y_test_pred[y_test_pred['y_pred'] == 1])}")


def stacking(alg, x_train_copy, y_train, ind):
    prediction = pd.DataFrame([0] * len(y_train), index=y_train.index, columns=["y_pred"])

    for i in ind:
        middle_pred = pd.DataFrame([0] * len(x_train_copy), index=x_train_copy.index, columns=["y_pred"])
        x_train_p, y_train_p = x_train_copy.drop(i, axis=0), y_train.drop(i, axis=0)
        x_test_p = x_train_copy.loc[i, :]

        # y_train_p = train["pm25mark"]
        # x_train_p = train.drop(labels=["pm25mark"], axis=1)
        # x_test_p = test.drop(labels=["pm25mark"], axis=1)

        # 选择分类器
        if alg == "xgboost":
            params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                      "max_depth": [10, 20, 30, 40, 50],
                      "n_estimators": [100, 300, 500, 700, 900],
                      "subsample": [0.1, 0.3, 0.5, 0.7, 1],
                      "gamma": [0.1, 0.3, 0.5, 0.7, 0.9],
                      "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9]}
            estimator = XGBClassifier(tree_method="hist")

        if alg == "lightgbm":
            params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                      "max_depth": [5, 10, 20, 30, 50],
                      "n_estimators": [100, 300, 500, 700, 900],
                      "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
                      "subsample": [0.1, 0.3, 0.5, 0.7, 1]}
            estimator = LGBMClassifier()

        estimator = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=1)
        estimator.fit(x_train_p, y_train_p)
        y_pred = estimator.predict(x_test_p)

        prediction["y_pred"][i] = y_pred
        # prediction = pd.concat([prediction, middle_pred], axis=0)
        # prediction.index = range(len(prediction))
    return prediction


def random_ind(num_stack, x_train, y_train):
    y_train = pd.DataFrame(y_train.values, columns=["y_train"], index=y_train.index)
    y_train_new = pd.DataFrame([0] * len(y_train), columns=["new"], index=y_train.index)
    ind = []
    for i in range(num_stack):
        data_index = x_train.index

        random_index = random.sample(list(data_index), int(0.2 * len(data_index)))
        x_train.drop(labels=list(random_index), axis=0, inplace=True)  # 删除已选中的索引
        y_train_new["new"][random_index] = y_train["y_train"][random_index]  # 保持与y_pred数量一致
        ind.append(random_index)
    return ind, y_train_new


if __name__ == "__main__":
    all_pred = pd.DataFrame()
    num_stack = 5

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    x_train_copy = deepcopy(x_train)

    # 将训练集分成五份
    ind, y_train_new = random_ind(num_stack, x_train, y_train)

    # xgboost
    single_pred_1 = stacking("xgboost", x_train_copy, y_train, ind)
    all_pred["pred1"] = single_pred_1["y_pred"]
    # lightgbm
    single_pred_2 = stacking("lightgbm", x_train_copy, y_train, ind)
    all_pred["pred2"] = single_pred_2["y_pred"]
    print(all_pred.head(10))
    print(len(all_pred))
    print(len(y_train_new))

    all_index = list(itertools.chain.from_iterable(ind))

    all_pred["y_train"] = y_train_new["new"]
    print(all_pred.head(10))

