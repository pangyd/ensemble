import pandas as pd
import random

pd.set_option("display.max_rows", None)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)   # 列名对齐
pd.set_option("display.width", 180)   # 横向最多显示多少个字符

#province = ["河北", "陕西", "山西", "山东", "河南", "四川", "江西"]
#province = ["山东"]

station_data = pd.read_csv("province_information.csv")

station_data = station_data.drop_duplicates(subset=["StationCode"], keep="first")

#station_data = station_data[station_data["Province"].isin(province)]
station_data = station_data.loc[~(station_data["PositionName"].str.contains("145")) &
                                ~(station_data["PositionName"].str.contains("微调")) &
                                ~(station_data["PositionName"].str.contains("新增20")) &
                                ~(station_data["PositionName"].str.contains("对照点")) &
                                ~(station_data["PositionName"].str.contains("金砖")) &
                                ~(station_data["PositionName"].str.contains("停运")) &
                                ~(station_data["PositionName"].str.contains("验收")) &
                                ~(station_data["PositionName"].str.contains("停用"))]
# print(len(station_data["StationCode"].unique()))
# 随机选取n个站点
station_data = station_data[station_data["StationCode"].isin(random.sample(list(station_data["StationCode"]), 200))]
station_data = station_data.sort_values(by="StationCode")
print(len(station_data))
station_data.index = range(len(station_data))
station_group = station_data.groupby(by="City")

# 按照城市分组
# city_l = station_data["City"].unique()
# station1 = station_data[station_data["City"].isin(city_l[:25])]
# station2 = station_data[station_data["City"].isin(city_l[25:50])]
# station3 = station_data[station_data["City"].isin(city_l[50:75])]
# station4 = station_data[station_data["City"].isin(city_l[75:])]
#
# city_group1 = station1.groupby(by="City")
# city_group2 = station2.groupby(by="City")
# city_group3 = station3.groupby(by="City")
# city_group4 = station4.groupby(by="City")

# print(station_data)

