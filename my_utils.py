import os
import pandas as pd
from itertools import chain

from sklearn.metrics import confusion_matrix, f1_score, precision_score


def cal_sta(sta, df):
    # try:
    if sta == 'sum':
        df = df.sum(axis=1)
    elif sta == 'max':
        df = df.max(axis=1)
    elif sta == 'min':
        df = df.min(axis=1)
    elif sta == 'mean':
        df = df.mean(axis=1)
    elif sta == 'std':
        df = df.std(axis=1)
    else:
        assert 1 == 2, "统计指标不符合标准，请检查字段名{}是否正确！！！".format(sta)
    return df

def get_xgb_features(xgb_model, xgb_feature_path='../out_features/xgb_features.csv'):
    """获取 xgb 的特征重要度"""
    feature_names = xgb_model.feature_name()
    feature_importances = xgb_model.feature_importance()
    df_xgb_features = pd.DataFrame({'feature': feature_names, 'scores': feature_importances})
    df_xgb_features = df_xgb_features.sort_values('scores', ascending=False)
    df_xgb_features.to_csv(xgb_feature_path, index=False)


def check_path(_path):
    """Check whether the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))


def params_append(list_params_left, list_param_right):
    if type(list_params_left) is not list:
        list_params_left = list(map(lambda p: list([p]), list_params_left))
    n_left = len(list_params_left)
    n_right = len(list_param_right)
    list_params_left *= n_right
    list_param_right = list(chain([[p] * n_left for p in list_param_right]))
    print('list_params_left is', list_params_left)
    print('list_parms_right', list_param_right)
    params = []
    for i in range(len(list_params_left)):
        params = [list_params_left[i]]
        # print(params)
        params.append(list_param_right[i])
    return params


def get_grid_params(search_params):

    keys = list(search_params.keys())
    values = list(search_params.values())
    grid_params = list()
    if len(keys) == 1:
        for value in values[0]:
            dict_param = dict()
            dict_param[keys[0]] = value
            grid_params.append(dict_param.copy())
        return grid_params
    list_params_left = values[0]
    for i in range(1, len(values)):
        list_param_right = values[i]
        list_params_left = params_append(list_params_left, list_param_right)
    print('New list_params_left is', list_params_left)
    for params in list_params_left:
        dict_param = dict()
        for i in range(len(keys)):
            dict_param[keys[i]] = params[i]
        grid_params.append(dict_param.copy())
    return grid_params


def feature_analyze(model, to_print=False, csv_path=None):
    """
    XGBOOST 模型特征重要性分析。
    Args:
        model: 训练好的 xgb 模型。
        to_print: bool, 是否输出每个特征重要性。
        csv_path: str, 保存分析结果到 csv 文件路径。
    """
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    # if to_print:
    #     print(''.join(fs))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            f.writelines("feature,score\n")
            f.writelines(fs)
    return feature_score


def evaluate(y_true, y_pre):
    """
    :param y_true:真实标签
    :param y_pre:预测标签
    :return:conf_mat:混淆矩阵 f1: pre_score:预测准确率
    """
    thres = 0.5
    y_pre = (y_pre > thres)
    conf_mat = confusion_matrix(y_true, y_pre, labels=[0, 1])
    f1 = f1_score(y_true, y_pre)
    pre_score = precision_score(y_true, y_pre)
    return conf_mat, f1, pre_score
