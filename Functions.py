# =============================================================================
# 优化算法用到的函数
# =============================================================================

# In[] 
from func_configs import *
import _pickle as cpickle
import pandas as pd
from Functions_2 import *
from scipy.spatial import distance_matrix
import sklearn.metrics as grade
import scipy.stats as s
from xgboost import XGBRegressor as XGBR  # XGboost
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
import statsmodels.api as sm


# In[] 保存数据
class save_read_data():
    def save(self, data=None, file='ga.pickle'):
        with open(file, "wb") as fout:
            cpickle.dump([data], fout)
        return None

    def read(self, file='ga.pickle'):
        # loading
        with open(file, "rb") as fin:
            data = cpickle.load(fin)
        return data[0]


# In[] 任务
class Task():
    def __init__(self):
        return None


# In[] 获取变量
def get_feature_00(x, name_x, y):
    # 示例数据矩阵
    data_matrix = x
    # 将数据矩阵转换成DataFrame
    df = pd.DataFrame(data_matrix, columns=name_x)
    df['target'] = y
    # 计算所有特征之间的相关系数矩阵
    correlation_matrix = df.corr(method='spearman')
    # 提取特征与目标变量的相关性
    correlation_with_target = correlation_matrix['target'].abs().sort_values(ascending=False)
    # 筛选出相关性大于0.6的特征对
    highly_correlated_pairs = (
        correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
        .stack()
        .sort_values(ascending=False)
        .reset_index())
    highly_correlated_pairs.columns = ['feature1', 'feature2', 'correlation']
    highly_correlated_pairs = highly_correlated_pairs[highly_correlated_pairs['correlation'] > 0.6]
    # 删除与目标变量相关性低的特征
    features_to_drop = []
    for _, row in highly_correlated_pairs.iterrows():
        feature1, feature2 = row['feature1'], row['feature2']
        if correlation_with_target.loc[feature1] < correlation_with_target.loc[feature2]:
            features_to_drop.append(feature1)
        else:
            features_to_drop.append(feature2)
    # 删除重复项
    features_to_drop = list(set(features_to_drop))
    # 最终删除特征
    # df.drop(columns=features_to_drop, inplace=True)
    num = []
    for i in features_to_drop:
        for j in range(len(name_x)):
            if i == name_x[j]:
                num += [j]
                break
    sur = np.setdiff1d([i for i in range(len(name_x))], num)

    return sur


# In[] 提取特征
def get_feature(x0, y0):
    # In[] 计算p值
    p_value = []
    #
    for n in range(x0.shape[1]):
        xn = x0[:, n]
        xn = sm.add_constant(xn)
        logit_model = sm.Logit(y0, xn)
        result = logit_model.fit()
        p_value += [result.pvalues[1]]
    p_value = np.array(p_value)

    # In[] 提取特征
    ind = np.where(p_value < 0.01)[0]

    return ind, p_value


# In[] 读取数据

def read_data():
    # 训练组数据 + 预测组数据
    file = inner_source_data if sampler_suffix == "" else sampled_data
    # 读取数据
    data = pd.read_csv(file).to_numpy().astype('float32')
    ind = np.where(np.isnan(data[:, 0]))[0]
    data = np.delete(data, ind, axis=0)
    # 第0列是 结果 y0, 其他列是数据 x0
    x0, y0 = data[:, 1:], data[:, 0]
    x00 = copy.deepcopy(x0)
    y00 = copy.deepcopy(y0)

    # 获取名字
    name = np.array(pd.read_csv(file, header=None))[0]
    name_x, name_y = name[1:], name[0]

    # 划分训练组和预测组
    x_train, y_train, x_test, y_test = training_test_gen(x0, y0, train_ratio=0.85)

    # 外部验证数据 
    # 读取数据
    data = pd.read_csv(outer_source_data).to_numpy().astype('float32')
    ind = np.where(np.isnan(data[:, 0]))[0]
    data = np.delete(data, ind, axis=0)
    x0, y0 = data[:, 1:], data[:, 0]
    x_validate, y_validate = copy.deepcopy(x0), copy.deepcopy(y0)

    # 保存数据
    data = [x00, y00, x_train, y_train, x_test, y_test, x_validate, y_validate, name_x, name_y]
    save_read_data().save(data, data_pickle)
    return data


# In[] 设置参数
class Load_Data():
    def __init__(self):
        # 读取数据
        # data=save_read_data().read(file='data.pickle')
        data = read_data()
        self.seed = 43

        # 使用 spearman 提取特征
        # x-> x00->原x, y->y00->原y name_x->name_x
        x, y, name_x = data[0], data[1], data[8]
        index = get_feature_00(x, name_x, y)
        # index -> 过滤完相关系数过大的列之后的数据index
        feature = Task()
        feature.index, feature.x_name = index, name_x[index]
        self.feature = feature

        # 保存结果 
        self.x0, self.y0, self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_validate, self.Y_validate = \
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        self.data = data

        # In[] 标准化数据
        #
        xn = mapminmax(self.X_train)
        self.standarlize = mapminmax(self.X_train)
        x_train = xn.apply(self.X_train)
        x_test = xn.apply(self.X_test)
        x_validate = xn.apply(self.X_validate)

        #
        y_train = copy.deepcopy(self.Y_train)
        y_test = copy.deepcopy(self.Y_test)
        y_validate = copy.deepcopy(self.Y_validate)

        #
        self.xn = xn
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_validate = x_validate
        self.y_validate = y_validate

        # 阈值
        self.threshold = 0.5
        return None
