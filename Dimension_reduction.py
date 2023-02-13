import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from torch.nn import Embedding

pd.set_option("display.max_columns", None)


data = pd.read_excel("train_2020/dataset/顺义新城.xlsx",index_col=0)
data.set_index(keys=["TimePoint"], inplace=True)
data.drop(labels=['City', 'StationName', 'StationCode'], axis=1, inplace=True)
print(data.head())

# PCA
pca = PCA(n_components=8)
pca_data = pca.fit_transform(data)
pca_data = round(pd.DataFrame(pca_data), 2)
print(pca_data.head())

# VarianceThreshold
vt = VarianceThreshold(threshold=0.5)
vt_data = vt.fit_transform(data)
vt_data = pd.DataFrame(vt_data)
print(vt_data.head())

# r2
result = pearsonr(data["O3"], data["SO2"])
print(result[0], result[1])

# embedding
tensor_data = torch.LongTensor(np.array(data))
emb = Embedding(num_embeddings=len(data), embedding_dim=5)
emb_np = emb(tensor_data).detach().numpy()
emb_data = pd.DataFrame(emb_np)
print(emb_data.head())


