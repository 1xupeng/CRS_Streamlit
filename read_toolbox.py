import copy
import matplotlib.pyplot as plt
from pathlib import Path
sampler_str = "RandomUnderSampler"
sampler_suffix = sampler_str
data_pickle = "current_origin_data.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_data.pickle"

name = ['AdaBoost', 'ANN', 'DT', 'ET', 'GBM', 'KNN', 'LightGBM', 'LR', 'RF', 'SVM', 'XGboost']
feature_pickle = "current_origin_feature.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_feature.pickle"
feature_pickle_names = [f"{model_name}_{feature_pickle}" for model_name in name]
top_lens = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
model_pickle_names = [f"__{model_name}_saved.pkl" for model_name in name]
data_pickle_name = "__"+data_pickle 
load_existed = True
feature_pickles = ["__" + n for n in feature_pickle_names]
streamlit_app_name = "CRS Mortality Predictor for ICU"

import pickle

# In[] 保存数据
class save_read_data():

    def read(self, file=data_pickle_name):
        return pickle.load(open(file, "rb"))


# In[] 设置参数
class Load_Data():
    def __init__(self,i=1):
        with open(feature_pickles[i], 'rb') as f:
            self.feature = pickle.load(f)

