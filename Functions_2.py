# =============================================================================
#   拟合用到的函数  
# =============================================================================


# In[ ] 导入库
import numpy as np
import sklearn.metrics as grade  #  mean_absolute_error mean_squared_error r2_score
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import copy
from func_configs import random_seed


# In[] 划分数据 训练组和预测组
def training_test_gen(x, y, train_ratio=0.7):
    # 样本数目
    all = x.shape[0]
    # 打乱数据集编号
    # all_index = np.arange(0, all)
    np.random.seed(random_seed)
    all_index = np.random.permutation(all)
    # 训练组编号
    train_index = all_index[:int((train_ratio) * all)]

    # 预测组编号
    #all_index=np.random.permutation(all)
    test_index = all_index[int(train_ratio * all):]

    # 检验组
    all_index = np.random.permutation(all)
    validate_index = all_index[int(train_ratio * all):]

    # 训练组
    x_train = x[train_index]
    y_train = y[train_index]
    # 检验组
    x_validate = x[validate_index]
    y_validate = y[validate_index]
    # 预测组
    x_test = x[test_index]
    y_test = y[test_index]

    return x_train, y_train, x_test, y_test


# In[] 归一化与反归一化
class mapminmax():
    def __init__(self, X, lb=-1, ub=1):
        self.Min = np.min(X, axis=0)  # X 为一列一个数据
        self.Max = np.max(X, axis=0)
        self.lb = lb
        self.ub = ub
        self.eps = 1e-16

    def apply(self, X):
        out = (X - self.Min) / (self.Max - self.Min + self.eps) * (self.ub - self.lb) + self.lb
        return out

    def apply_index(self,X, index):
        out = (X - self.Min[index]) / (self.Max[index] - self.Min[index] + self.eps) * (self.ub - self.lb) + self.lb
        return out

    def reversed(self, X, index):
        out = (X - self.lb) / (self.ub - self.lb + self.eps) * (self.Max[index] - self.Min[index]) + self.Min[index]
        return out

    def reverse_multi(self,X:np.ndarray, indexes):
        result = X.copy()
        feature_len = len(X[0])
        for i in range(feature_len):
            result[:,i] = self.reversed(X[:,i],indexes[i])
        return result


# In[] 计算评价指标 
# 参考网址： https://blog.csdn.net/sdu_hao/article/details/103533115
def cal_score_00000(y_real, y_predict, threshold=0.5):
    v = []
    v += [accuracy_score(y_real, y_predict > threshold)]  # 0：准确率，这个最重要
    v += [precision_score(y_real, y_predict > threshold)]  # 1
    v += [recall_score(y_real, y_predict > threshold)]  # 2
    v += [f1_score(y_real, y_predict > threshold)]  # 3
    v += [roc_auc_score(y_real, y_predict)]  # 4
    v += [confusion_matrix(y_real, y_predict > threshold)]  # 5
    v += [classification_report(y_real, y_predict > threshold)]  # 6

    y_true, y_pred = copy.deepcopy(y_real), copy.deepcopy(y_predict) > threshold
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # 计算灵敏度和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # 使用sklearn计算F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    v += [v[4], sensitivity, specificity, f1]  # 7~10

    return v
