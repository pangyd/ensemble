import logging
import pandas as pd
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


# logging.basicConfig(filename="/suncere/pyd/xgb/all_feature_data/xgb_bagging.log", filemode="a",
#                     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#data = features.data
data = pd.read_csv("D://data_judgement/out_features/test长三角地区.csv", index_col=0)
data.index = range(len(data))
# print(data.columns)
data.drop(labels=["城市", "站点名称", "站点编号", "pm10mark"], axis=1, inplace=True)
data = data.replace([-np.inf, np.inf], np.nan).dropna(axis=0)

# 测试集
time1 = data[data["时间"] >= "2020-08-01 23:00:00"]
test_f = data[data["时间"].isin(list(time1["时间"]))]

data.drop(labels=["时间"], axis=1, inplace=True)
test_f.drop(labels=["时间"], axis=1, inplace=True)
data = data.drop(test_f.index, axis=0)
test_f.index = range(len(test_f))

# 随机选取
# normal_data = data[data["pm25mark"] == 0]
# exception_data = data[data["pm25mark"] == 1]
# normal_index = data[data["pm25mark"] == 0].index
# new_index = random.sample(list(normal_index), int(0.1 * len(normal_data)))
# normal_new = data.loc[new_index, :]
# data = pd.concat([normal_new, exception_data])
# data.index = range(len(data))

y = data["pm25mark"]
x = data.drop(labels=["pm25mark"], axis=1)

num_model = 5