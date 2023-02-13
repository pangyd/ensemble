import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import flaml
import supervised
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


def flaml_automl(x_train, y_train):
    params = {"time_budget": 60,
              "task": "classification",
              "estimator_list": ["xgboost", "lgbm", "rf"],
              "n_splits": 5,
              "eval_method": "cv",
              "random_seed": 123}

    automl = flaml.automl.AutoML()
    automl.fit(x_train, y_train, **params)
    return automl


def mljar_automl(x_train, y_train):
    automl = supervised.automl.AutoML(total_time_limit=60,
                                      mode="Perform",
                                      train_ensemble=True,
                                      eval_metric="mse",
                                      algorithms=["Xgboost", "LightGBM", "Random Forest"],
                                      ml_task="regression")
    automl.fit(x_train, y_train)
    return automl


def xgb(x_train, y_train):
    params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
              "max_depth": [10, 20, 30, 40, 50],
              "n_estimators": [100, 300, 500, 700, 900],
              "subsample": [0.1, 0.3, 0.5, 0.7, 1],
              "gamma": [0.1, 0.3, 0.5, 0.7, 0.9],
              "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9]}
    estimator = XGBClassifier(tree_method="hist")
    estimator = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=1)
    estimator.fit(x_train, y_train)
    return estimator


def lgbm(x_train, y_train):
    params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
              "max_depth": [5, 10, 20, 30, 50],
              "n_estimators": [100, 300, 500, 700, 900],
              "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
              "subsample": [0.1, 0.3, 0.5, 0.7, 1]}
    estimator = LGBMClassifier()
    estimator = RandomizedSearchCV(estimator=estimator, param_distributions=params, n_iter=1)
    estimator.fit(x_train, y_train)
    return x_train, y_train


def random_forest(x_train, y_train):
    params = {"max_depth": [10, 20, 30, 40, 50],
              "n_estimators": [50, 100, 150, 200],
              "min_samples_split": [1, 2, 3]}
    estimator = RandomForestClassifier(n_jobs=-1)
    estimator = RandomizedSearchCV(param_distributions=params, estimator=estimator, n_iter=10, cv=5)
    estimator.fit(x_train, y_train)
    return x_train, y_train


def diy_stacking(alg, x_train, y_train, ind, y_test, preddiction_astest):
    """
    ind: 训练集分成五分的索引的列表
    """
    prediction_astrain = pd.DataFrame([0] * len(y_train), index=y_train.index, columns=["y_pred"])

    for k, i in enumerate(ind):
        x_train_p, y_train_p = x_train.drop(i, axis=0), y_train.drop(i, axis=0)
        x_test_p = x_train.loc[i, :]

        # 选择分类器
        if alg == "xgboost":
            estimator = xgb(x_train_p, y_train_p)

        if alg == "lightgbm":
            estimator = lgbm(x_train_p, y_train_p)

        if alg == "flaml":
            estimator = flaml_automl(x_train_p, y_train_p)

        if alg == "mljar":
            estimator = mljar_automl(x_train_p, y_train_p)

        # 保存模型
        with open("{}_alg.dat".format(alg), "wb") as model:
            pickle.dump(estimator, model)

        # 预测第五份训练集
        y_train_pred = estimator.predict(x_test_p)
        prediction_astrain["y_pred"][i] = y_train_pred

        # 预测测试集
        preddiction_astest["y_pred_{}".format(i+1)] = estimator.predict(y_test)

    return prediction_astrain, preddiction_astest


def final_training(x_train, x_test, y_train, y_test):
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    with open("stacking_lr.dat", "wb") as model:
        pickle.dump(lr, model)

    y_pred = lr.predict(x_test)

    mse = mean_squared_error(y_pred, y_test)
    mae = mean_absolute_error(y_pred, y_test)
    return mse, mae


def random_ind(num_stack, x_train, y_train):
    """将训练集平均分成五份"""
    y_train = pd.DataFrame(y_train.values, columns=["y_train"], index=y_train.index)
    y_train_new = pd.DataFrame([0] * len(y_train), columns=["y_train_new"], index=y_train.index)
    ind = []   # 每次划分的索引

    for i in range(num_stack):
        data_index = x_train.index
        random_index = random.sample(list(data_index), int(0.2 * len(data_index)))
        x_train.drop(labels=list(random_index), axis=0, inplace=True)  # 删除已选中的索引
        y_train_new["y_train_new"][random_index] = y_train["y_train"][random_index]  # 保持与y_pred数量一致
        ind.append(random_index)
    return ind, y_train_new


if __name__ == "__main__":
    all_pred_astrain = pd.DataFrame()
    all_pred_astest = pd.DataFrame()
    alg_list = ["xgboost", "lightgbm", "flaml", "mljar"]
    num_stack = 5

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    x_train_copy = deepcopy(x_train)

    # 将训练集分成五份
    ind, y_train_new = random_ind(num_stack, x_train, y_train)

    prediction_astest = pd.DataFrame(index=y_test.index)
    for alg in alg_list:
        single_pred, prediction_astest = diy_stacking(alg, x_train_copy, y_train, ind, y_test, prediction_astest)
        all_pred_astrain["pred_xgb"] = single_pred["y_pred"]
        all_pred_astest["pred_xgb"] = prediction_astest.sum(axis=1).apply(lambda x: 1 if x > 2 else 0)  # 投票

    # # xgboost
    # single_pred_1, prediction_astest_xgb = diy_stacking("xgboost", x_train_copy, y_train, ind, y_test, prediction_astest)
    # all_pred_astrain["pred_xgb"] = single_pred_1["y_pred"]
    # all_pred_astest["pred_xgb"] = prediction_astest_xgb.sum(axis=1).apply(lambda x: 1 if x > 2 else 0)   # 投票
    #
    # # lightgbm
    # single_pred_2, prediction_astest_lgbm = diy_stacking("lightgbm", x_train_copy, y_train, ind, y_test, prediction_astest)
    # all_pred_astrain["pred_lgbm"] = single_pred_2["y_pred"]
    # all_pred_astest["pred_lgbm"] = prediction_astest_xgb.sum(axis=1).apply(lambda x: 1 if x > 2 else 0)

    # 构造新的训练集和测试集
    # all_pred_astrain["y_train"] = y_train
    # all_pred_astest["y_test"] = y_test

    # 用新数据集进行训练和预测
    final_training(all_pred_astrain, y_train_new, all_pred_astest, y_test)

    all_index = list(itertools.chain.from_iterable(ind))


