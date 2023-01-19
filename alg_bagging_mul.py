from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
import joblib
from ensemble_predata import *
from sklearn.ensemble import BaggingClassifier

# data1 = print(int(os.path.getsize("D://data_judgement/out_features/real_data.csv")) / 1024 / 1024)


def bagging_meta(x, y):
    x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    clf = BaggingClassifier(base_estimator=XGBClassifier(), n_estimators=7, )
    clf.fit(x_train, y_train)

    with open("meta_bagging.pkl", "wb") as bag:
        joblib.dump(clf, bag)

    y_pred = clf.predict(x_test)
    print("真实值中异常值的数量:{}".format(len(y_test[y_test == 1])))
    print("预测集中异常值的数量:{}".format(len(y_pred[y_pred == 1])))
    print("异常值预测准确的数量：{}，真实值中预测准确的比例：{},"
          " 预测集中预测准确的比例：{}".format(len(y_test[(y_test == y_pred) & (y_test == 1)]),
                                           len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_test[y_test == 1]),
                                           len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_pred[y_pred == 1])))


# [xgboost, lightgbm, random_forest, logistic_regressor]
def multi_algorithm(algorithm):
    if algorithm == "xgboost":
        estimator = XGBClassifier()
        params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                  "max_depth": [5, 10, 20, 30, 50],
                  "n_estimators": [100, 300, 500, 700, 900],
                  "gamma": [0.1, 0.3, 0.5, 0.7, 0.9],
                  "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
                  "subsample": [0.1, 0.3, 0.5, 0.7, 1]}
    if algorithm == "lgbm":
        estimator = LGBMClassifier()
        params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                  "max_depth": [5, 10, 20, 30, 50],
                  "n_estimators": [100, 300, 500, 700, 900],
                  "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
                  "subsample": [0.1, 0.3, 0.5, 0.7, 1]}
    if algorithm == "rf":
        estimator = RandomForestClassifier(random_state=123)
        params = {"max_depth": [5, 10, 20, 30, 50],
                  "n_estimators": [100, 300, 500, 700, 900]}
    if algorithm == "lr":
        estimator = LogisticRegression()
    return estimator, params


# 三个xgboost
def bagging_train():
    for al, i in zip(alg, range(num_model)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        if al == "lr":
            estimator, _ = multi_algorithm(algorithm=al)
        else:
            est, params = multi_algorithm(algorithm=al)
            # estimator = GridSearchCV(estimator=xgb, param_grid=xgb_params, n_jobs=-1, cv=5)
            estimator = RandomizedSearchCV(estimator=est, param_distributions=params, n_iter=30, n_jobs=-1, cv=5)

        estimator.fit(x_train, y_train)

        print("第{}轮最好的参数:{}".format(i+1, estimator.best_params_))
        print("第{}轮最好的模型:{}".format(i+1, estimator.best_estimator_))
        logging.info(f"第{i+1}轮最好的参数:{estimator.best_params_}")
        logging.info(f"第{i+1}轮最好的模型:{estimator.best_estimator_}")

        with open("/suncere/pyd/xgb/all_feature_data/{}_bagging.pkl".format(al), "wb") as xgb_estimator:
            joblib.dump(estimator, xgb_estimator)

        y_pred = estimator.predict(x_test)
        print("第{}轮:真实值中异常值的数量:{}".format(i+1, len(y_test[y_test == 1])))
        print("第{}轮:预测集中异常值的数量:{}".format(i+1, len(y_pred[y_pred == 1])))
        print("第{}轮：异常值预测准确的数量：{}，真实值中预测准确的比例：{},"
              " 预测集中预测准确的比例：{}".format(i+1,
                                               len(y_test[(y_test == y_pred) & (y_test == 1)]),
                                               len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_test[y_test == 1]),
                                               len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_pred[y_pred == 1])))
        logging.info(f"训练时测试集的准确率为:")
        logging.info(f"第{i+1}轮:真实值中异常值的数量:{len(y_test[y_test == 1])}")
        logging.info(f"第{i+1}轮:预测集中异常值的数量:{len(y_pred[y_pred == 1])}")
        logging.info(f"第{i+1}轮：异常值预测准确的数量：{len(y_test[(y_test == y_pred) & (y_test == 1)])}，"
                     f"真实值中预测准确的比例：{len(y_test[(y_test == y_pred) & (y_test == 1)]) / len(y_test[y_test == 1])}")

# bagging
def bagging_test_1(test_f):
    y_test_pred = pd.DataFrame(test_f["pm25mark"], index=range(len(test_f)), columns=["pm25mark"])
    y_test_pred["xgb_pred"] = 0
    test_f = test_f.drop(labels=["pm25mark"], axis=1)
    for j in range(num_model):
        with open("/suncere/pyd/xgb/all_feature_data/xgb_bagging_{}.pkl".format(j+1), "rb") as xgb_model:
            xgb_model = joblib.load(xgb_model)
        y_pred = xgb_model.predict(test_f)
        y_test_pred["xgb_pred_{}".format(j+1)] = y_pred
    #    y_test_pred["xgb_pred_{}".format(j + 1)] = y_test_pred["xgb_pred_{}".format(j+1)].astype(int)
    print(y_test_pred.head())

    for k in range(len(y_test_pred)):
        n = 0
        for m in range(num_model):
            n += y_test_pred["xgb_pred_{}".format(m+1)][k]
        if n >= int(num_model / 2) + 1:
            y_test_pred["xgb_pred"][k] = 1
        else:
            y_test_pred["xgb_pred"][k] = 0

    print("真实值中异常值的数量:{}".format(len(y_test_pred[y_test_pred["pm25mark"] == 1])))
    print("预测集中异常值的数量:{}".format(len(y_test_pred[y_test_pred["xgb_pred"] == 1])))
    print("异常值预测准确的数量：{}，真实值中预测准确的比例：{}, 预测集中预测准确的比例：{}".format(len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]),
                                                                len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]) / len(
                                                                y_test_pred[y_test_pred["pm25mark"] == 1]),
                                                                len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]) / len(
                                                                y_test_pred[y_test_pred["xgb_pred"] == 1])))
    #logging.info(f"真实预测集的预测结果:")
    #logging.info(f"真实值中异常值的数量:{len(y_test_pred[y_test_pred['pm25mark'] == 1])}")
    #logging.info(f"预测集中异常值的数量:{len(y_test_pred[y_test_pred['xgb_pred'] == 1])}")
    #logging.info(f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['pm25mark'] == y_test_pred['xgb_pred']) & (y_test_pred['pm25mark'] == 1)])}，"
    #             f"真实值中预测准确的比例：{len(y_test_pred[(y_test_pred['pm25mark'] == y_test_pred['xgb_pred']) & (y_test_pred['pm25mark'] == 1)]) / len(y_test_pred[y_test_pred['pm25mark'] == 1])}")


def bagging_test_2(test_f):
    y_test_pred = pd.DataFrame(test_f["pm25mark"], index=range(len(test_f)), columns=["pm25mark"])
    y_test_pred["xgb_pred"] = 0
    test_f = test_f.drop(labels=["pm25mark"], axis=1)
    for j in range(num_model):
        with open("/suncere/pyd/xgb/all_feature_data/xgb_bagging_{}.pkl".format(j+1), "rb") as xgb_model:
            xgb_model = joblib.load(xgb_model)
        y_pred = xgb_model.predict(test_f)

        y_test_pred["xgb_pred_{}".format(j + 1)] = y_pred
        pred_index = y_test_pred[y_test_pred["xgb_pred_{}".format(j + 1)] == 1].index
        # 每个模型预测出的异常值写入最终的预测列
        y_test_pred["xgb_pred"][pred_index] = 1

        test_f.drop(labels=pred_index, axis=0, inplace=True)

    print("真实值中异常值的数量:{}".format(len(y_test_pred[y_test_pred["pm25mark"] == 1])))
    print("预测集中异常值的数量:{}".format(len(y_test_pred[y_test_pred["xgb_pred"] == 1])))
    print("异常值预测准确的数量：{}，真实值中预测准确的比例：{}, 预测集中预测准确的比例：{}".format(
        len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]),
        len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]) / len(
            y_test_pred[y_test_pred["pm25mark"] == 1]),
        len(y_test_pred[(y_test_pred["pm25mark"] == y_test_pred["xgb_pred"]) & (y_test_pred["pm25mark"] == 1)]) / len(
            y_test_pred[y_test_pred["xgb_pred"] == 1])))


if __name__ == "__main__":
    alg = ["xgboost", "lgbm", "rf", "lr"]
    num_model = len(alg)

    bagging_train()
