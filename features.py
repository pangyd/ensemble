import pandas as pd
import numpy as np
import time
from my_utils import cal_sta
import test_data

# 聚合统计特征指标
stas = ['sum', 'max', 'min', 'mean', 'std']
days = [1, 3, 5, 7, 9, 11, 25, 27, 31, 33, 35, 37]


# bias
def get_bias_features(pro, year):
    df_18 = pd.read_csv("../datasets/长三角地区-{}.csv".format(year))
    df_18.columns = ["城市", "站点名称", "站点编号", "时间", "SO2", "SO2mark", "NO2", "NO2mark",
                     "O3", "O3mark", "CO", "COmark", "pm10", "pm10mark", "pm25", "pm25mark"]
    df_18_pm25_pm10 = df_18.drop(columns=["SO2", "SO2mark", "NO2", "NO2mark",
                                          "O3", "O3mark", "CO", "COmark"])
    if pro == '非对照点':
        df = df_18_pm25_pm10.loc[
            (~df_18_pm25_pm10["站点名称"].str.contains("145")) &
            (~df_18_pm25_pm10["站点名称"].str.contains("微调")) &
            (~df_18_pm25_pm10["站点名称"].str.contains("新增20")) &
            (~df_18_pm25_pm10["站点名称"].str.contains("对照点"))]
    else:
        df = df_18_pm25_pm10.loc[
            (~df_18_pm25_pm10["站点名称"].str.contains("145")) &
            (~df_18_pm25_pm10["站点名称"].str.contains("微调")) &
            (~df_18_pm25_pm10["站点名称"].str.contains("新增20"))]
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
        df["{}_other_value".format(poll)].replace([0, np.nan], df["{}_other_value".format(poll)].mean(), inplace=True)
        df["{}_dev".format(poll)] = (df[poll] - df["{}_other_value".format(poll)]) / df[
            "{}_other_value".format(poll)]  # 差值与其他站点平均值的比
        df["{}_sort".format(poll)] = df.groupby(["城市", "时间"])[poll].rank(method="min", ascending=False)

    df = df.sort_values(by=["城市", "站点编号", "时间"], ascending=[True, True, True])

    for poll in pollutions:
        for col in cols:
            for day in days:
                print("当前构造特征为{}_days{}_{}".format(poll, day, col))
                df["{}_day{}_{}".format(poll, day, col)] = df.groupby('站点编号')["{}_{}".format(poll, col)].shift(day * 24)
                df["{}_day{}_{}".format(poll, day, col)].fillna(df.groupby('站点编号')["{}_day{}_{}".format(poll, day, col)].mean(), inplace=True)
    # time cost
    for hour in range(1, 6):
        df["pm25_hour{}_pre".format(hour)] = df.groupby('站点编号')["pm25"].shift(hour)

    for hour in range(1, 6):
        df["pm25_hour{}_pre_other".format((hour))] = df.groupby('站点编号')["pm25_other_value"].shift(hour)

    return df


def get_deep_features():
    df_quyv_model = test_data.data
    df_quyv_model.index = range(len(df_quyv_model))
    print(len(df_quyv_model.columns))
    # fetch data
    # df_quyv_model = df_quyv.loc[(df_quyv["pm25mark"] == "RM") | (df_quyv["pm25mark"].isna())]
    # df_quyv_model = df_quyv_model.loc[(df_quyv_model["pm10mark"] == "RM") | (df_quyv_model["pm10mark"].isna())]
    # df_quyv_model = df_quyv_model.loc[(df_quyv_model["pm25"] != (-99)) & (df_quyv_model["pm10"] != (-99))]
    # pollutions = ['pm25', 'pm10']
    # for poll in pollutions:
    #     df_quyv_model.loc[df_quyv_model["{}mark".format(poll)] != 'RM', "{}mark".format(poll)] = 0
    #     df_quyv_model.loc[df_quyv_model["{}mark".format(poll)] == 'RM', "{}mark".format(poll)] = 1
    #
    #     df_quyv_model[poll].replace([0, np.nan], df_quyv_model[poll].mean(), inplace=True)
    #     df_quyv_model[poll + '_dev'].replace([0, np.nan], df_quyv_model[poll + '_dev'].mean(), inplace=True)

    df_quyv_model = df_quyv_model.sort_values(by=["城市", "站点编号", "时间"], ascending=[True, True, True])
    """
    （1）标记正常数据为0，异常数据为1；将0和空值替换为平均值
    （2）分别计算差值与pm25、pm10的比值
    （3）计算列表天数与目标站点的数字特征
    """
    """
    abs:history bias features
    diff:pm10、pm25与同城差值的差值、历史差值相关特征构造
    dev:pm10、pm25的偏差、占比、历史差值相关特征构造
    per:
    """
    # df_tem = pd.DataFrame()
    cols = ['abs', 'diff', 'dev', 'per']
    for col in cols:
        if col == 'abs':
            df_quyv_model["pm10_pm25_{}".format(col)] = abs(df_quyv_model["pm10"] - df_quyv_model["pm25"])
            df_quyv_model["pm10_pm25_{}_pm25".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm25"]  # 目标站点pm10、pm25差值与pm25的比
            df_quyv_model["pm10_pm25_{}_pm10".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm10"]  # 目标站点pm10、pm25差值与pm10的比
        elif col == 'diff':
            df_quyv_model["pm10_pm25_{}".format(col)] = abs(
                df_quyv_model["pm10_{}".format(col)] - df_quyv_model["pm25_{}".format(col)])
            df_quyv_model["pm10_pm25_{}_pm25".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm25"]
            df_quyv_model["pm10_pm25_{}_pm10".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm10"]
        elif col == 'dev':
            df_quyv_model["pm10_pm25_{}".format(col)] = abs(
                df_quyv_model["pm10_{}".format(col)] - df_quyv_model["pm25_{}".format(col)])
            df_quyv_model["pm10_pm25_{}_pm25".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm25_{}".format(col)]
            df_quyv_model["pm10_pm25_{}_pm10".format(col)] = df_quyv_model["pm10_pm25_{}".format(col)] / \
                                                             df_quyv_model["pm10_{}".format(col)]
        else:
            df_quyv_model["pm10_pm25_sort"] = abs(
                df_quyv_model["pm10_sort"] - df_quyv_model["pm25_sort"])
            df_quyv_model["pm25_diff_per"] = df_quyv_model["pm25_diff"] / df_quyv_model["pm25"]
            df_quyv_model["pm10_pm25_per"] = df_quyv_model["pm25"] / df_quyv_model["pm10"]

        for day in days:
            # print("当前构造特征为pm10_pm25_{}_hisday{}_diff".format(col, day))
            # 提取历史数据中对应时间点的差值
            df_quyv_model["pm10_pm25_{}_day{}_diff".format(col, day)] = \
                abs(df_quyv_model.groupby('站点编号')["pm10_pm25_{}".format(col)].shift(day * 24))
            # df_quyv_model["pm10_pm25_{}_day{}_diff".format(col, day)].fillna(df_quyv_model["pm10_pm25_{}_day{}_diff".format(col, day)].mean(), inplace=True)
            # 计算当前时刻的差值（pm25-pm10）与历史时刻的差值
            df_quyv_model["pm10_pm25_{}_hisday{}_diff".format(col, day)] = abs(
                df_quyv_model["pm10_pm25_{}".format(col)] - df_quyv_model["pm10_pm25_{}_day{}_diff".format(col, day)])

        for sta in stas:
            df_t = df_quyv_model[
                ["pm10_pm25_%s_hisday1_diff" % col, "pm10_pm25_%s_hisday3_diff" % col,
                 "pm10_pm25_%s_hisday5_diff" % col, "pm10_pm25_%s_hisday7_diff" % col,
                 "pm10_pm25_%s_hisday9_diff" % col, "pm10_pm25_%s_hisday11_diff" % col,
                 "pm10_pm25_%s_hisday25_diff" % col, "pm10_pm25_%s_hisday27_diff" % col,
                 "pm10_pm25_%s_hisday31_diff" % col, "pm10_pm25_%s_hisday33_diff" % col,
                 "pm10_pm25_%s_hisday35_diff" % col, "pm10_pm25_%s_hisday37_diff" % col
                 ]]
            df_quyv_model["pm25_pm10_{}_his_diff_{}".format(col, sta)] = cal_sta(sta, df_t)
        # df_tem = pd.concat([df_tem, df_diff_count])
    # df_quyv_model = pd.merge(df_quyv_model, df_tem, how='left', on='站点编号')
    # print(df_quyv_model.columns)
    return df_quyv_model


def get_dynamic_features(df_quyv_model):
    pollutions = ['pm25', 'pm10']
    cols = ['dev', 'sort', 'diff']
    for poll in pollutions:
        for col in cols:
            for day in days:
                # print("当前构造特征为{}_his{}{}_difference".format(poll, col, day))
                df_quyv_model["{}_his{}{}_difference".format(poll, col, day)] = \
                    abs(df_quyv_model["{}_{}".format(poll, col)] -
                        df_quyv_model["{}_day{}_{}".format(poll, day, col)])
    df_quyv_model["pm25_hisdevdiff_agg"] = 0

    lists = [[5, 1.5, 1.5], [5, 10, 1.2], [10, 15, 1],
             [15, 20, 0.9], [20, 25, 0.7], [25, 35, 0.6],
             [35, 50, 0.35], [50, 70, 0.25], [70, 100, 0.2],
             [100, 1.5, 0.18]]
    for list in lists:
        if list[0] == 5 and list[1] == 1.5:
            df_quyv_model.loc[
                (df_quyv_model["pm25_other_value"] <= 5) & (
                        (abs(df_quyv_model["pm25_hisdev1_difference"]) >= 1.5) | (
                        abs(df_quyv_model["pm25_hisdev37_difference"]) >= 1.5)) &
                ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= 1.5) | (
                        abs(df_quyv_model["pm25_hisdev35_difference"]) >= 1.5)) &
                ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= 1.5) | (
                        abs(df_quyv_model["pm25_hisdev33_difference"]) >= 1.5)) &
                ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= 1.5) | (
                        abs(df_quyv_model["pm25_hisdev31_difference"]) >= 1.5)) &
                ((abs(df_quyv_model["pm25_hisdev9_difference"]) >= 1.5) | (
                        abs(df_quyv_model["pm25_hisdev27_difference"]) >= 1.5)) &
                ((abs(df_quyv_model["pm25_hisdev11_difference"]) >= 1.5) | (abs(
                    df_quyv_model["pm25_hisdev25_difference"]) >= 1.5)), "pm25_hisdevdiff_agg"] = 1
#            df_quyv_model.loc[
#                (df_quyv_model["pm25_other_value"] <= 5) & (
#                        (abs(df_quyv_model["pm25_hisdev1_difference"]) >= 1.5)) &
#                        ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= 1.5)) &
#                        ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= 1.5)) &
#                        ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= 1.5)), "pm25_hisdevdiff_agg"] = 1

        elif list[0] == 100 and list[1] == 1.5:
#            df_quyv_model.loc[
#                (df_quyv_model["pm25_other_value"] > 100) & (
#                        (abs(df_quyv_model["pm25_hisdev1_difference"]) >= 0.18)) &
#                        ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= 0.18)) &
#                        ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= 0.18)) &
#                        ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= 0.18)), "pm25_hisdevdiff_agg"] = 1

            df_quyv_model.loc[
                (df_quyv_model["pm25_other_value"] > 100) & ((abs(df_quyv_model["pm25_hisdev1_difference"]) >= 0.18) | (
                        abs(df_quyv_model["pm25_hisdev37_difference"]) >= 0.18)) &
                ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= 0.18) | (
                        abs(df_quyv_model["pm25_hisdev35_difference"]) >= 0.18)) &
                ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= 0.18) | (
                        abs(df_quyv_model["pm25_hisdev33_difference"]) >= 0.18)) &
                ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= 0.18) | (
                        abs(df_quyv_model["pm25_hisdev31_difference"]) >= 0.18)) &
                ((abs(df_quyv_model["pm25_hisdev9_difference"]) >= 0.18) | (
                        abs(df_quyv_model["pm25_hisdev27_difference"]) >= 0.18)) &
                ((abs(df_quyv_model["pm25_hisdev11_difference"]) >= 0.18) | (abs(
                    df_quyv_model["pm25_hisdev25_difference"]) >= 0.18)), "pm25_hisdevdiff_agg"] = 1
        else:
#            df_quyv_model.loc[
#                (df_quyv_model["pm25_other_value"] > list[0]) &
#                (df_quyv_model["pm25_other_value"] <= list[1]) & (
#                    (abs(df_quyv_model["pm25_hisdev1_difference"]) >= list[2])) &
#                        ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= list[2])) &
#                        ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= list[2])) &
#                        ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= list[2])), "pm25_hisdevdiff_agg"] = 1

            df_quyv_model.loc[
                (df_quyv_model["pm25_other_value"] > list[0]) &
                (df_quyv_model["pm25_other_value"] <= list[1]) & (
                        (abs(df_quyv_model["pm25_hisdev1_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev37_difference"]) >= list[2])) &
                ((abs(df_quyv_model["pm25_hisdev3_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev35_difference"]) >= list[2])) &
                ((abs(df_quyv_model["pm25_hisdev5_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev33_difference"]) >= list[2])) &
                ((abs(df_quyv_model["pm25_hisdev7_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev31_difference"]) >= list[2])) &
                ((abs(df_quyv_model["pm25_hisdev9_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev27_difference"]) >= list[2])) &
                ((abs(df_quyv_model["pm25_hisdev11_difference"]) >= list[2]) | (
                        abs(df_quyv_model["pm25_hisdev25_difference"]) >= list[2])), "pm25_hisdevdiff_agg"] = 1
    return df_quyv_model


def split_bias(df_quyv_model):
    df_quyv_model["pm25_dev_des"] = 0
    df_quyv_model.loc[
        (df_quyv_model["pm25_other_value"] <= 5) &
        (abs(df_quyv_model["pm25_dev"]) >= 2) &
        (abs(df_quyv_model["pm10_pm25_dev"]) >= 2), "pm25_dev_des"] = 1
    i, j = 5, 1.5
    count = 0
    while count <= 9:
        if count >= 5 and count <= 6:
            tem = i
            i += 10
            j -= 0.05
        elif count == 7:
            tem = i
            i += 20
        elif count == 8:
            tem = i
            i += 30
        elif count == 9:
            tem = i
            i += 50
        else:
            tem = i
            i += 5
            j -= 0.3
        df_quyv_model.loc[
            (df_quyv_model["pm25_other_value"] > tem) &
            (df_quyv_model["pm25_other_value"] <= i) &
            (abs(df_quyv_model["pm25_dev"]) >= j) &
            (abs(df_quyv_model["pm10_pm25_dev"]) >= j), "pm25_dev_des"] = 1
        count += 1

    df_quyv_model.loc[(df_quyv_model["pm25_other_value"] > 150) &
                      (abs(df_quyv_model["pm25_dev"]) >= 0.18) &
                      (abs(df_quyv_model["pm10_pm25_dev"]) >= 0.18), "pm25_dev_des"] = 1

    return df_quyv_model


def pm25_bias(df_quyv_model):
    df_quyv_model["pm10_pm25_difference_des"] = 0
    lists = [[10, 15, 15], [15, 25, 20], [25, 40, 25],
             [40, 60, 30], [60, 100, 35]]
    for list in lists:
        df_quyv_model.loc[
            (df_quyv_model["pm25_other_value"] > list[0]) &
            (df_quyv_model["pm25_other_value"] <= list[1]) &
            (abs(df_quyv_model["pm10_pm25_diff"]) >= list[2]),
            "pm10_pm25_difference_des"] = 1
    df_quyv_model.loc[
        (df_quyv_model["pm25_other_value"] > 100) &
        (abs(df_quyv_model["pm10_pm25_diff"]) >= 40),
        "pm10_pm25_difference_des"] = 1
    cols = ['dev', 'sort', 'diff']
    for col in cols:
        for sta in stas:
            df_t = df_quyv_model[
                ["pm25_his%s1_difference" % col, "pm25_his%s3_difference" % col, "pm25_his%s5_difference" % col,
                 "pm25_his%s7_difference" % col
                 ]]
            df_quyv_model["pm25_{}_{}".format(col, sta)] = cal_sta(sta, df_t)

    return df_quyv_model


def five_hours_features(df_quyv_model):
    # 构造排名差值规则
    df_quyv_model["pm25_sort_des"] = 0
    df_quyv_model.loc[df_quyv_model["pm25_sort_sum"] >= 8, "pm25_sort_des"] = 1

    # 所有规则统计描述
    df_quyv_model["all_des_sum"] = df_quyv_model[["pm25_hisdevdiff_agg", "pm25_dev_des",
                                                  "pm10_pm25_difference_des", "pm25_sort_des"]].sum(axis=1)
    for hour in range(5, 0, -1):
        df_quyv_model["pm25_pre{}_diff".format(hour)] = \
            abs(df_quyv_model["pm25"] - df_quyv_model["pm25_hour{}_pre".format(hour)])
        df_quyv_model["pm25_pre{}_diff_other".format(hour)] = \
            abs(df_quyv_model["pm25_other_value"] - df_quyv_model["pm25_hour{}_pre_other".format(hour)])

    for sta in stas:
        df_t = df_quyv_model[
            ["pm25_pre5_diff", "pm25_pre4_diff",
             "pm25_pre3_diff", "pm25_pre2_diff",
             "pm25_pre1_diff"]]
        df_quyv_model["pm25_hour_{}".format(sta)] = cal_sta(sta, df_t)

    for sta in stas:
        df_quyv_model["pm25_hour_{}_pm25".format(sta)] = \
            df_quyv_model["pm25_hour_{}".format(sta)] / df_quyv_model["pm25"]

    # neighbor stations
    for sta in stas:
        df_t = df_quyv_model[
            ["pm25_pre5_diff_other", "pm25_pre4_diff_other",
             "pm25_pre3_diff_other", "pm25_pre2_diff_other",
             "pm25_pre1_diff_other"]]
        df_quyv_model["pm25_pre_hour_{}_other".format(sta)] = cal_sta(sta, df_t)

    for sta in stas:
        df_quyv_model["pm25_pre_hour_{}_other_diff".format(sta)] = \
            df_quyv_model["pm25_hour_{}".format(sta)] - \
            df_quyv_model["pm25_pre_hour_{}_other".format(sta)]

    df_quyv_model["station_attribute"] = 0
    df_quyv_model.loc[df_quyv_model["站点名称"].str.contains("对照点"), "station_attribute"] = 1
    df_quyv_model["pm25_pm10_at_one_time"] = 0
    df_quyv_model.loc[(df_quyv_model["pm25_other_value"] <= 5) &
                      (abs(df_quyv_model["pm25_dev"]) >= 2) &
                      (abs(df_quyv_model["pm10_dev"]) >= 2),
                      "pm25_pm10_at_one_time"] = 1
    lists = [[5, 10, 1.5, 1], [10, 15, 1, 0.7], [15, 25, 0.8, 0.6],
             [25, 35, 0.5, 0.4], [35, 50, 0.35, 0.25], [50, 70, 0.25, 0.2],
             [70, 100, 0.2, 0.2]]
    for list in lists:
        df_quyv_model.loc[(df_quyv_model["pm25_other_value"] > list[0]) &
                          (df_quyv_model["pm25_other_value"] <= list[1]) &
                          (abs(df_quyv_model["pm25_dev"]) >= list[2]) &
                          (abs(df_quyv_model["pm10_dev"]) >= list[3]),
                          "pm25_pm10_at_one_time"] = 1
    df_quyv_model.loc[(df_quyv_model["pm25_other_value"] > 100) &
                      (abs(df_quyv_model["pm25_dev"]) >= 0.18) &
                      (abs(df_quyv_model["pm10_dev"]) >= 0.15),
                      "pm25_pm10_at_one_time"] = 1
    df_quyv_model["时间"] = pd.to_datetime(df_quyv_model["时间"])
    df_quyv_model = df_quyv_model.loc[df_quyv_model["时间"] > pd.to_datetime("201801080200")]
    return df_quyv_model


#if __name__ == '__main__':
    # year = 2020
    # pros = ['对照点', '非对照点']
    # for pro in pros:
    #     df = get_bias_features(pro=pro, year=year)
    #     if pro == '非对照点':
    #         df.to_csv('../features/长三角地区{}年非对照点第一次处理后数据0911.csv'.format(year))
    #     else:
    #         df = df.loc[df["站点名称"].str.contains("对照点")]
    #         df.to_csv("../features/长三角地区{}年对照点第一次处理后数据0911.csv".format(year))
    # time.sleep(60)
df = get_deep_features()
#    print("###" * 10)
#    print("get_deep_features run completely!")
df = get_dynamic_features(df)
#    print("###" * 10)
#    print("get_dynamic_features run completely!")
df = split_bias(df)
#    print("###" * 10)
#    print("split_bias run completely!")
df = pm25_bias(df)
#    print("###" * 10)
#    print("pm25_bias run completely!")
df = five_hours_features(df)
#    print(df.columns)
#    print("###" * 10)
data = df.dropna()

# 样本不均衡-随机减少正常数据的比例
import random
rate = 0.15
normal_data = data[data["pm25mark"] == 0]
exception_data = data[data["pm25mark"] == 1]
index_list = list(normal_data.index)
list1 = random.sample(index_list, int(rate * len(normal_data)))
normal_data = normal_data.loc[list1, :]
data = pd.concat([normal_data, exception_data])
data.index = range(len(data))
print("正常数据数量：", len(normal_data))
print("异常数据数量", len(exception_data))

#df.to_csv("/suncere/pyd/xgb/all_feature_data/2022_7.csv", index=False)
#    print("Congratulations!!! All done!!! ^_^ ^_^ ^_^")
    # pd.set_option("display.max_rows", None)
    # df = pd.read_csv("sichuan_all_feature.csv")
    # print(df.isna().sum())



