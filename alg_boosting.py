from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from ensemble_predata import *


# 创建权重数列
weight = pd.DataFrame([0.5]*len(x), index=x.index, columns=["weight"])

xgb_params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
              "max_depth": [10, 20, 30, 40, 50],
              "n_estimators": [100, 300, 500, 700, 900],
              "subsample": [0.1, 0.3, 0.5, 0.7, 1],
              "gamma": [0.1, 0.3, 0.5, 0.7, 0.9],
              "reg_lambda": [0.1, 0.3, 0.5, 0.7, 0.9]}


def boosting_train(x, y):
    for i in range(num_model):
        xgb = XGBClassifier(tree_method='hist')

        estimator = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=20, cv=5)

        estimator.fit(x, y)

        print("第{}轮最好的参数:{}".format(i+1, estimator.best_params_))
        print("第{}轮最好的模型:{}".format(i+1, estimator.best_estimator_))
        logging.info(f"第{i+1}轮最好的参数:{estimator.best_params_}")
        logging.info(f"第{i+1}轮最好的模型:{estimator.best_estimator_}")

        with open("xgb_boosting_{}.dat".format(i+1), "wb") as xgb_estimator:
            pickle.dump(estimator, xgb_estimator)

        y_pred = estimator.predict(x)

        print("第{}轮:真实值中异常值的数量:{}".format(i + 1, len(y[y == 1])))
        print("第{}轮:预测集中异常值的数量:{}".format(i + 1, len(y_pred[y_pred == 1])))
        print("第{}轮：异常值预测准确的数量：{}，真实值中预测准确的比例：{}, 预测集中预测准确的比例：{}".format(i + 1,
                                                         len(y[(y == y_pred) & (y == 1)]),
                                                         len(y[(y == y_pred) & (y == 1)]) / len(y[y == 1]),
                                                         len(y[(y == y_pred) & (y == 1)]) / len(y_pred[y_pred == 1])))
        logging.info(f"训练时测试集的准确率为:")
        logging.info(f"第{i + 1}轮:真实值中异常值的数量:{len(y[y == 1])}")
        logging.info(f"第{i + 1}轮:预测集中异常值的数量:{len(y_pred[y_pred == 1])}")
        logging.info(f"第{i + 1}轮：异常值预测准确的数量：{len(y[(y == y_pred) & (y == 1)])}，"
                     f"真实值中预测准确的比例：{len(y[(y == y_pred) & (y == 1)]) / len(y[y == 1])}")
        logging.info(f"第{i + 1}轮：异常值预测准确的数量：{len(y[(y == y_pred) & (y == 1)])}，"
                     f"预测集中预测准确的比例：{len(y[(y == y_pred) & (y == 1)]) / len(y_pred[y_pred == 1])}")

        # 修改样本权重 -- 降低预测正确的权重、提高预测错误的权重
        weight["weight_new_{}".format(i+1)] = 0
        weight["weight_new_{}".format(i+1)][y[y == y_pred].index] = 0.75
        weight["weight_new_{}".format(i+1)][y[y != y_pred].index] = 0.25

        # 按照权重来选取数据集
        next_fault_index = random.sample(weight[weight["weight_new_{}".format(i+1)] == 0.75].index, int(0.9 * len(weight)))
        next_true_index = random.sample(weight[weight["weight_new_{}".format(i+1)] == 0.25].index, int(0.5 * len(weight)))
        x = x[next_true_index + next_fault_index]
        y = y[next_true_index + next_fault_index]


def boosting_test():
    y_test_pred = pd.DataFrame(test_f["pm25mark"], index=test_f.index, columns=["y_test"])
    y_test = test_f["pm25mark"]
    x_test = test_f.drop(labels=["pm25mark"], axis=1)
    # 同样以投票形式做出决策
    for j in range(num_model):
        with open("xgb_boosting_{}".format(j+1), "rb") as xgb_model:
            xgb_boost = pickle.load(xgb_model)
        y_pred = xgb_boost.predict(x_test)
        y_test_pred["y_pred_{}".format(j+1)] = y_pred

    for k in range(len(test_f)):
        n = 0
        for m in range(num_model):
            n += y_test_pred["y_pred_{}".format(m+1)][k]
        # if n >= int(num_model / 2) + 1:
        if n >=1:
            y_test_pred["y_pred"][k] = 1
        else:
            y_test_pred["y_pred"][k] = 0

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
    logging.info(f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)])}，"
                 f"真实值中预测准确的比例：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)]) / len(y_test_pred[y_test_pred['y_test'] == 1])}")
    logging.info(
        f"异常值预测准确的数量：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)])}，"
        f"预测集中预测准确的比例：{len(y_test_pred[(y_test_pred['y_test'] == y_test_pred['y_pred']) & (y_test_pred['y_test'] == 1)]) / len(y_test_pred[y_test_pred['y_pred'] == 1])}")


if __name__ == "__main__":
    boosting_train(x, y)
    boosting_test()