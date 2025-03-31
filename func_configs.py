from pathlib import Path

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

from imblearn.under_sampling import TomekLinks, RandomUnderSampler, ClusterCentroids, InstanceHardnessThreshold, \
      NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, \
      CondensedNearestNeighbour, NeighbourhoodCleaningRule
from imblearn.over_sampling import ADASYN, RandomOverSampler, KMeansSMOTE, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, \
      SMOTEN

random_seed = 43

sampler_str = "RandomUnderSampler"
sampler_suffix = sampler_str

samplers = {
      # down-sampling
      "TomekLinks": TomekLinks(),
      "RandomUnderSampler": RandomUnderSampler(random_state=random_seed),
      "ClusterCentroids": ClusterCentroids(random_state=random_seed),
      "InstanceHardnessThreshold": InstanceHardnessThreshold(random_state=random_seed),
      "NearMiss": NearMiss(),
      "EditedNearestNeighbours": EditedNearestNeighbours(),
      "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours(),
      "AllKNN": AllKNN(),
      "OneSidedSelection": OneSidedSelection(random_state=random_seed),
      "CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=random_seed),
      "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(),

      # up-sampling
      "ADASYN": ADASYN(random_state=random_seed),
      "RandomOverSampler": RandomOverSampler(random_state=random_seed),
      "KMeansSMOTE": KMeansSMOTE(random_state=random_seed),
      "SMOTE": SMOTE(random_state=random_seed),
      "BorderlineSMOTE": BorderlineSMOTE(random_state=random_seed),
      "SVMSMOTE": SVMSMOTE(random_state=random_seed),
      "SMOTEN": SMOTEN(random_state=random_seed),
}
pic_path = Path(f'./picture/{sampler_suffix}_/')
if not pic_path.parent.exists():
    pic_path.parent.mkdir()
if not pic_path.exists():
    pic_path.mkdir()

inner_source_data = "DataCleaning/data/CRS_IV.csv"
outer_source_data = './DataCleaning/data/CRS_eicu.csv'


cleaned_data = "DataCleaning/OriginData/pured_iv_data.csv"
sampled_data = f'./DataCleaning/data/11/1/iv_1_cbh_{sampler_suffix}_.csv'  # smote data

data_pickle = "current_origin_data.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_data.pickle"
feature_pickle = "current_origin_feature.pickle" if sampler_suffix == "" else f"current_{sampler_suffix}_feature.pickle"

model_info = {
      "ET": {"index": 0, "model_name": "ET", "model": ET(random_state=random_seed), "top_len": 6},
      "ANN": {"index": 1, "model_name": "ANN", "model": ANN(random_state=random_seed), "top_len": 6},
      "LR": {"index": 2, "model_name": "LR", "model": LR(random_state=random_seed), "top_len": 7},


      "RF": {"index": 3, "model_name": "RF", "model": RF(random_state=random_seed), "top_len": 6},

      "GBM": {"index": 4, "model_name": "GBM", "model": GBM(random_state=random_seed), "top_len": 8},

      "KNN": {"index": 5, "model_name": "KNN", "model": KNN(n_neighbors=5, algorithm="ball_tree"), "top_len": 4},
      "LightGBM": {"index": 6, "model_name": "LightGBM", "model": LightGBM(random_state=random_seed), "top_len": 11},
      "DT": {"index": 7, "model_name": "DT", "model": DT(random_state=random_seed), "top_len": 2},
      "AdaBoost": {"index": 8, "model_name": "AdaBoost", "model": AdaBoost(random_state=random_seed), "top_len": 11},
      "SVM": {"index": 9, "model_name": "SVM", "model": SVM(probability=True, random_state=random_seed), "top_len": 11},
      "XGboost": {"index": 10, "model_name": "XGboost", "model": XGboost(), "top_len": 9},
}
name = [model_name for model_name in model_info]
seeds = [random_seed for n in name]
# name = ['AdaBoost', 'ANN', 'DT', 'ET', 'GBM', 'KNN', 'LightGBM', 'LR', 'RF', 'SVM', 'XGboost']
MM = [model_info[model_name]["model"] for model_name in model_info]
top_lens = [model_info[model_name]["top_len"] for model_name in model_info]
feature_pickle_names = [f"{model_name}_{feature_pickle}" for model_name in name]
model_pickle_names = [f"__{model_name}_saved.pkl" for model_name in name]
data_pickle_name = "__"+data_pickle
feature_pickles = ["__" + n for n in feature_pickle_names]
load_existed = False

