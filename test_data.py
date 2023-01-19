import numpy as np
import pandas as pd
import time
import warnings
from sqlalchemy import create_engine
from urllib.parse import quote
import station_information
from multiprocessing import Pool
from functools import partial

pd.set_option("display.max_columns", None)

warnings.filterwarnings("ignore")

# 连接数据库
user = "sa"
password = "sun123!@"
host = "10.10.204.102:18710"
database = "EnvDataChina"
engine1 = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)

years = [2021, 2022]

# 污染物编号
code = ["104", "105"]
feature_dict = {"104": "pm10", "105": "pm25"}

# 监测数据特征列表
feature_list = []
for key in feature_dict:
    feature_list.append(feature_dict[key])

# 聚合统计特征指标
stas = ['sum', 'max', 'min', 'mean', 'std']
days = [1, 3, 5, 7, 9, 11, 25, 27, 31, 33, 35, 37]

start_time = time.time()

def get_data(city_group, all_city_data):
    for label1, group1 in city_group:   # 按照城市顺序进行合并
        for index in group1.index:   # 按照每个城市的所有站点进行合并
            single_station_data = pd.DataFrame()
            print("正在读取{}站点信息....".format(group1["StationCode"][index]))
            for year in years:
                if year == 2022:
                    year_data = pd.date_range(start="2022-01-01 01:00:00", end="2022-07-01 00:00:00", freq="H")
                    year_data = pd.DataFrame(year_data, columns=["时间"])
                else:
                    year_data = pd.date_range(start="2021-11-25 01:00:00", end="2022-01-01 00:00:00", freq="H")
                    year_data = pd.DataFrame(year_data, columns=["时间"])

                # 如果表不存在则跳过
                try:
                    # 单个站点
                    sq = "select * from Air_h_{}_{}_App where PollutantCode='104' or  PollutantCode='105'".format(year, group1["StationCode"][index])
                    df = pd.read_sql(sq, engine1)  # 单个站点
                except Exception as result:
                    print("该站点不存在数据")
                else:
                    # 判断表是否为空表或缺少污染物，是则退出内循环
                    if df.empty | ("104" not in df["PollutantCode"].values) | ("105" not in df["PollutantCode"].values):
                        print("该表为空表or缺失污染物")
                        break
                    # 按照污染物分组，按照时间合并
                    grouped = df.groupby(by="PollutantCode")
                    label_list = []
                    for label, group in grouped:  # label:[str]
                        label_list.append(label)
                        if label in feature_dict.keys():
                            group = group[["TimePoint", "MonValue", "Mark"]]
                            # print(group)
                            # 有异常标识且异常表示不为RM的污染物浓度赋值nan
                            # print(len(group[group["Mark"] == ""].index))
                            group["MonValue"][~group["Mark"].isin(["", "RM", "H"])] = np.nan
                            # print(group[group["Mark"] == "RM"].index)
                            # print(group[group["Mark"] == 1].index)
                            group = group.rename(columns={"MonValue": "{}".format(feature_dict[label]),
                                                          "Mark": "{}mark".format(feature_dict[label]),
                                                          "TimePoint": "时间"})
                            year_data = year_data.merge(group, on="时间", how="left")
                            # 异常值为1，非异常值为0
                            year_data.loc[year_data["{}mark".format(feature_dict[label])] != "RM", "{}mark".format(feature_dict[label])] = 0
                            year_data.loc[year_data["{}mark".format(feature_dict[label])] == "RM", "{}mark".format(feature_dict[label])] = 1
                            year_data.loc[year_data["{}".format(feature_dict[label])].isin([-99, 0.0]), "{}".format(feature_dict[label])] = np.nan
                            # group["Mark"][group["Mark"] == "RM"] = 1
                            # group["Mark"][group["Mark"] != "RM"] = 0

                            # [PM2.5, PM10]都×1000
                            if feature_dict[label] in ["pm25", "pm10"]:
                                year_data[feature_dict[label]] = year_data[feature_dict[label]] * 1000
                        # 若label不在污染物列表内，则删除该组
                        else:
                            useless_label = grouped.get_group(label).index
                            df = df.drop(useless_label)

                    # 拼接每年数据
                    single_station_data = single_station_data.append(year_data)
            # 加入站点编号、城市
            if single_station_data.empty:
                pass
            else:
                _ = pd.Series([group1["StationCode"][index]] * len(single_station_data))
                single_station_data.insert(1, "站点编号", _)
                _ = pd.Series([group1["City"][index]] * len(single_station_data))
                single_station_data.insert(1, "城市", _)
                _ = pd.Series([group1["PositionName"][index]] * len(single_station_data))
                single_station_data.insert(1, "站点名称", _)
                # 拼接每个站点的数据
                all_city_data = all_city_data.append(single_station_data)
    return all_city_data


def multi_pool():
    all_city_data = pd.DataFrame()
    all_city_data = get_data(station_information.station_group, all_city_data)
    all_city_data[["pm25", "pm10"]] = all_city_data[["pm25", "pm10"]].interpolate(method="linear", limit_direction="forward")
    all_city_data[["pm25", "pm10"]] = all_city_data[["pm25", "pm10"]].interpolate(method="linear", limit_direction="backward")
    all_city_data[["pm25mark", "pm10mark"]] = all_city_data[["pm25mark", "pm10mark"]].astype(int)
    print("数据下载完毕！！！")
    return all_city_data


def original():
    all_city_data = pd.DataFrame()
    all_city_data = get_data(station_information.station_group, all_city_data)
    all_city_data = all_city_data[all_city_data["pm25"] > 25]
    all_city_data[["pm25mark", "pm10mark"]] = all_city_data[["pm25mark", "pm10mark"]].astype(int)
    print("数据下载完毕！！！")
    return all_city_data


def get_bias_features():

    df = multi_pool()

    pollutions = ['pm25', 'pm10']
    cols = ['dev', 'sort', 'diff']
    '''
    (1)计算除当前站点外所有污染物数值的和，再求平均值；
    (2)计算当前站点的污染物数值与平均值的差；
    (3)差值与其他站点平均值的比；
    (4)按城市时间站点编号从小到大排名
    (5)位移列表所列天数和6小时值
    '''

    for poll in pollutions:
        df["cs_{}".format(poll)] = df.groupby(["城市", "时间"])[poll].transform("sum")
        df["cs_{}_except".format(poll)] = df["cs_{}".format(poll)] - df[poll]

    df["station_counts"] = df.groupby(["城市", "时间"])["站点名称"].transform("count")
    df = df.loc[df["station_counts"].values > 1]

    for poll in pollutions:
        df["{}_other_value".format(poll)] = (df["cs_{}_except".format(poll)] / (df["station_counts"] - 1))  # 同城其他站点平均值
        df["{}_diff".format(poll)] = df[poll] - df["{}_other_value".format(poll)]  # 目标站点和平均值的差值
        df["{}_other_value".format(poll)].fillna(df["{}_other_value".format(poll)].mean(), inplace=True)
        df["{}_dev".format(poll)] = (df[poll] - df["{}_other_value".format(poll)]) / df[
            "{}_other_value".format(poll)]  # 差值与其他站点平均值的比
        df["{}_sort".format(poll)] = df.groupby(["城市", "时间"])[poll].rank(method="min", ascending=False)

    df = df.sort_values(by=["城市", "站点编号", "时间"], ascending=[True, True, True])

    for poll in pollutions:
        for col in cols:
            for day in days:
                # print("当前构造特征为{}_days{}_{}".format(poll, day, col))
                df["{}_day{}_{}".format(poll, day, col)] = df.groupby('站点编号')["{}_{}".format(poll, col)].shift(day * 24)
                # df["{}_day{}_{}".format(poll, day, col)].fillna(df["{}_day{}_{}".format(poll, day, col)].mean(), inplace=True)
    # time cost
    for hour in range(1, 6):
        df["pm25_hour{}_pre".format(hour)] = df.groupby('站点编号')["pm25"].shift(hour)
        # df["pm25_hour{}_pre".format(hour)].fillna(df["pm25_hour{}_pre".format(hour)].mean(), inplace=True)

    for hour in range(1, 6):
        df["pm25_hour{}_pre_other".format((hour))] = df.groupby('站点编号')["pm25_other_value"].shift(hour)
        # df["pm25_hour{}_pre_other".format((hour))].fillna(df["pm25_hour{}_pre_other".format((hour))].mean(), inplace=True)

    return df


"""
1.污染物值=nan - 先不理会；
2.mark=0占多数；
3.原始数据中缺失某些时间点的值，用线性插值进行补齐；
4.pm25和pm10中小于25的值先搁置，特征全部处理完成再删除
5.考虑归一化
"""

# 把六大污染物<=25的值转换成nan
# data["pm25"][data["pm25"] <= 25] = np.nan
# data["pm10"][data["pm10"] <= 25] = np.nan

# 获取第一次处理后的数据
data = get_bias_features()
#data.to_csv("first_process.csv")
#print("有数据的站点数量:", len(np.unique(list(data["站点编号"]))))

# 原始污染数据
# data = original()
#data.to_csv("/suncere/pyd/xgb/8province_data/2021_2022_pm25_pm10.csv")

end_time = time.time()
print("一共所花费的时间为:{}s".format(end_time-start_time))
