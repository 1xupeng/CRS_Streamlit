import pandas as pd
import numpy as np
import shap
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(csv_path):
    data_pd = pd.read_csv(csv_path, index_col=0)
    feature = data_pd.drop(['label'], axis=1)
    feature_list = feature.columns.tolist()
    label = data_pd['label']

    # 初始化标准化器
    scaler = StandardScaler()

    # 对特征进行标准化
    scaled_features = scaler.fit_transform(feature)
    feature_train, feature_test, label_train, label_test = train_test_split(scaled_features,label, test_size=0.2)
    return feature_train, label_train, feature_test, label_test,feature_list

def get_class_weight(data_train, label_train):
    # return {0:0.2,
    #         1:0.8}
    return {0: len(label_train[label_train == 1]) / len(label_train),
            1: len(label_train[label_train == 0]) / len(label_train)}

def train_model(data_train,label_train,feature_test, label_test):
    # 初始化和训练随机森林模型
    ml_model = RandomForestClassifier(class_weight=get_class_weight(data_train,label_train))
    ml_model.fit(data_train, label_train)

    # 预测并计算准确率
    test_pred = ml_model.predict(feature_test)
    print(f"Accuracy: {accuracy_score(label_test, test_pred)}")

    return ml_model

def show_shap(rf, input_data,feature_list, save_folder):
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(input_data)
    explanation = shap.Explanation(values=shap_values[...,1], base_values=explainer.expected_value, data=input_data,feature_names=feature_list)

    # GLOBAL
    # 绘制summary plot 二分类
    shap.summary_plot(shap_values, input_data,feature_names=feature_list, plot_type="bar",show=False)

# =============================================================================
#     # 保存SHAP summary plot为图片
#     plt.savefig(str(Path(save_folder)/'shap_summary_plot.png'))
#     plt.close()
# =============================================================================

    # 构建 SHAP Explanation 对象
    # 生成并保存 Beeswarm plot
    shap.summary_plot(explanation, plot_type="dot",show=False)
# =============================================================================
#     plt.savefig(str(Path(save_folder)/'shap_beeswarm_plot.png'))
#     plt.close()
# =============================================================================

    # 保存为 HTML
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[...,1], input_data,feature_names=feature_list)
    shap.save_html(str(Path(save_folder)/'shap_values.html'),force_plot)

    # FOCAL
    # visualize the first prediction's explanation
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[0, :,1],feature_names=feature_list)
    shap.save_html(str(Path(save_folder)/'single_values.html'),force_plot)

    # 绘制并保存第一个特征 dependence plot
    shap.plots.scatter(explanation[:,feature_list[0]], color=shap_values[:,0,1], hist=False,show=False)
# =============================================================================
#     plt.savefig(str(Path(save_folder)/f'dependence_plot_{feature_list[0]}.png'))
#     plt.clf()  # 清除当前figure
# =============================================================================

    # 绘制并保存第一个case waterfall
    shap.waterfall_plot(explanation[0],show=False)
# =============================================================================
#     plt.savefig(str(Path(save_folder)/f'waterfall_{feature_list[0]}.png'))
#     plt.clf()  # 清除当前figure
# =============================================================================
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       pvalue
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    log10_p = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)  # Computes log(10) of p-values
    return 10 ** log10_p


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)
def main():
    csv_path = r'.\train_numeric_feature.csv'
    save_folder =  r'.\Example'
    feature_train, label_train, feature_test, label_test, feature_list = load_data(csv_path)
    ml_model = train_model(feature_train,label_train,feature_test, label_test)
    show_shap(ml_model, feature_train,feature_list, save_folder)
