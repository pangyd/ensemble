import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
import warnings
from supervised.automl import AutoML
import random
import features

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#data = pd.read_csv("/suncere/pyd/xgb/8province_data/original_data.csv", index_col=0)
data = features.data
print(data.columns)
data["时间"] = pd.to_datetime(data["时间"])
#data["month"] = pd.DatetimeIndex(data["时间"]).month
data.drop(["城市", "站点名称", "站点编号", "时间", "pm25mark"], axis=1, inplace=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#normal_data = data[data["pm25mark"] == 0]
#exception_data = data[data["pm25mark"] == 1]
#print(len(normal_data))
#print(len(exception_data))

# 对正常数据进行随机采样
#index_list = list(normal_data.index)
#list1 = random.sample(index_list, int(0.85 * len(normal_data)))
#normal_data.drop(list1, axis=0, inplace=True)
#data = pd.concat([normal_data, exception_data])
#data.index = range(len(data))
#print(data.head(5))

y = data["pm10mark"]
x = data.drop(labels=["pm10mark"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# print(y["pm25mark"].value_counts(normalize=True))
# print(len(y[y == 1]))

# 降采样
# ran = RandomUnderSampler(random_state=42)
# x_train, y_train = ran.fit_resample(x, y)

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=123)
#
print("异常值的数量", len(y_train[y_train == 1]))
print("正常值的数量", len(y_train[y_train == 0]))

automl = AutoML(total_time_limit = int(60*60*0.5),
                algorithms = ["Xgboost", "LightGBM"],
                ml_task = "binary_classification",
                mode = "Compete",
                train_ensemble = True,
                eval_metric = "accuracy",
                n_jobs = -1,
                random_state = 123)

automl.fit(x_train, y_train, sample_weight=None, cv=None)
# with open("mljar_detect_pm25_origin_24h.dat", "wb") as file:
#     pickle.dump(automl, file)

# joblib.dump(estimator, "D:/数据异常判别/save_models/my_xgb.pkl")

#estimator = pickle.load(open("/suncere/pyd/xgb/8province_data/flaml_detect_pm25_8pallfea70per.dat", "rb"))
y_pred = automl.predict(x_test)

print("真实值中异常值的数量", len(y_test[y_test == 1]))
print("预测集中异常值的数量", len(y_pred[y_pred == 1]))
print("异常值预测准确的数量：{}，真实值中预测准确的比例：{}".format(len(y_test[(y_test == y_pred) & (y_test == 1)]),
                                    len(y_test[(y_test == y_pred) & (y_test == 1)])/len(y_test[y_test == 1])))
print("异常值预测准确的数量：{}，预测值中预测准确的比例：{}".format(len(y_test[(y_test == y_pred) & (y_test == 1)]),
                                    len(y_test[(y_test == y_pred) & (y_test == 1)])/len(y_pred[y_pred == 1])))


#print(data[0].shape)
