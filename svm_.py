from sklearn import svm
import pandas as pd


data = pd.read_excel("顺义新城.xlsx", index_col=0)
data.set_index(keys=["TimePoint"], inplace=True)
data.drop(labels=['City', 'StationName', 'StationCode'], axis=1, inplace=True)
print(data.head())

svc = svm.SVC(C=0.1, kernel="linear")

svc.fit(data, data['SO2'])
