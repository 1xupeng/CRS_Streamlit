from pathlib import Path

# 模型相关导入
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.neighbors import KNeighborsClassifier as KNN
from lightgbm import LGBMClassifier as LightGBM
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from xgboost import XGBClassifier as XGboost

# 采样相关导入
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, ClusterCentroids, InstanceHardnessThreshold, \
      NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, \
      CondensedNearestNeighbour, NeighbourhoodCleaningRule
from imblearn.over_sampling import ADASYN, RandomOverSampler, KMeansSMOTE, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, \
      SMOTEN

random_seed = 116

# 采样器设置 - 必须与你的缓存文件名匹配
sampler_str = "RandomUnderSampler"
sampler_suffix = sampler_str

# 文件夹路径设置
pic_path = Path(f'./picture/{sampler_suffix}_/')
if not pic_path.parent.exists():
    pic_path.parent.mkdir()
if not pic_path.exists():
    pic_path.mkdir()

# 原始数据路径（由于 load_existed=True，这些路径在运行时不会被触发）
inner_source_data = "DataCleaning/data/CRS_IV.csv"
outer_source_data = './DataCleaning/data/CRS_eicu.csv'
cleaned_data = "DataCleaning/OriginData/pured_iv_data.csv"
sampled_data = f'./DataCleaning/data/11/1/iv_1_cbh_{sampler_suffix}_.csv'

# 缓存文件命名逻辑
# 这里的命名必须与你上传的文件 __current_RandomUnderSampler_data.pickle 等一致
data_pickle = "current_origin_data.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_data.pickle"
feature_pickle = "current_origin_feature.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_feature.pickle"

# 模型配置字典
model_info = {
    "ET": {"index": 0, "model_name": "ET", "model": ET(random_state=random_seed), "top_len": 6},
    "RF": {"index": 1, "model_name": "RF", "model": RF(random_state=random_seed), "top_len": 6},
    "LR": {"index": 2, "model_name": "LR", "model": LR(random_state=random_seed), "top_len": 7},
    "ANN": {"index": 3, "model_name": "ANN", "model": ANN(random_state=random_seed), "top_len": 6},
    "GBM": {"index": 4, "model_name": "GBM", "model": GBM(random_state=random_seed), "top_len": 8},
    "KNN": {"index": 5, "model_name": "KNN", "model": KNN(n_neighbors=5, algorithm="ball_tree"), "top_len": 4},
    "LightGBM": {"index": 6, "model_name": "LightGBM", "model": LightGBM(random_state=random_seed), "top_len": 11},
    "DT": {"index": 7, "model_name": "DT", "model": DT(random_state=random_seed), "top_len": 2},
    "AdaBoost": {"index": 8, "model_name": "AdaBoost", "model": AdaBoost(random_state=random_seed), "top_len": 11},
    "SVM": {"index": 9, "model_name": "SVM", "model": SVM(probability=True, random_state=random_seed), "top_len": 11},
    "XGboost": {"index": 10, "model_name": "XGboost", "model": XGboost(), "top_len": 9},
}

# 自动生成文件名列表
name = [model_name for model_name in model_info]
MM = [model_info[model_name]["model"] for model_name in model_info]
top_lens = [model_info[model_name]["top_len"] for model_name in model_info]

# 重点：这里的前缀 "__" 必须与你文件夹下的文件名完全一致
feature_pickle_names = [f"{model_name}_{feature_pickle}" for model_name in name]
model_pickle_names = [f"__{model_name}_saved.pkl" for model_name in name]
data_pickle_name = "__" + data_pickle
feature_pickles = ["__" + n for n in feature_pickle_names]

# --- 关键修改：强制加载已存在的缓存，不读取 CSV ---
load_existed = True