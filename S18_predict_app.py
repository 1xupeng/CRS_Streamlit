import sys
from pathlib import Path

# 获取当前脚本所在的绝对路径
current_dir = Path(__file__).resolve().parent
# 将该路径添加到系统路径中，这样 Python 就能找到 func_configs.py 了
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import numpy as np
import pandas as pd
import shap.plots
import os
import pickle
import copy
import matplotlib.pyplot as plt
import streamlit as st

# 导入校准器，保留校准逻辑
from sklearn.calibration import CalibratedClassifierCV

from read_toolbox import *
from func_configs import (
    sampled_data, feature_pickle_names, name, MM, top_lens, 
    model_pickle_names, data_pickle_name, feature_pickles, 
    load_existed, sampler_suffix
)

def prepare_model(i: int):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[i], "rb"))
    else:
        P = Load_Data()
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[i])
        pickle.dump(feature, open(feature_pickles[i], "wb"))

    index, x_name = feature.index[:], feature.x_name[:]
    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"

    top_len = top_lens[i]
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]

    if load_existed:
        with open(model_pickle_names[i], 'rb') as f:
            model = pickle.load(f)
    else:
        model_raw = copy.deepcopy(MM[i])
        model_raw = model_raw.fit(x_train, y_train)
        model = CalibratedClassifierCV(model_raw, method="sigmoid", cv=None, ensemble=False)
        model.fit(x_train, y_train)
        with open(model_pickle_names[i], "wb") as f:
            pickle.dump(model, f)

    return model, x_name


def draw_force_plot(i: int, sample: list):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[i], "rb"))
    else:
        P = Load_Data()
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[i])
        pickle.dump(feature, open(feature_pickles[i], "wb"))

    index, x_name = feature.index[:], feature.x_name[:]
    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"

    top_len = top_lens[i]
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]

    if load_existed:
        with open(model_pickle_names[i], 'rb') as f:
            model = pickle.load(f)
    else:
        model_raw = copy.deepcopy(MM[i])
        model_raw = model_raw.fit(x_train, y_train)
        model = CalibratedClassifierCV(model_raw, method="sigmoid", cv=None, ensemble=False)
        model.fit(x_train, y_train)
        with open(model_pickle_names[i], "wb") as f:
            pickle.dump(model, f)

    def model4shap(x):
        return model.predict_proba(x)[:, 1]

    raw_sample = np.array(sample).reshape(1, -1)
    sample_scaled = P.standarlize.apply_index(raw_sample, index)
    predict_prob = model.predict_proba(sample_scaled)[:, 1]

    plt.figure()
    model_test_explainer = shap.Explainer(model4shap, masker=x_train, feature_names=x_name, seed=P.seed)
    model_test_shap = model_test_explainer(sample_scaled)[0]

    # 同步 S18app 的逻辑：设置显示数据的保留位数为 2 位
    model_test_shap.display_data = np.around(P.standarlize.reversed(sample_scaled, index), decimals=2)

    shap.plots.force(model_test_shap, feature_names=x_name, matplotlib=True, show=False)

    literal_feature_name = "".join([char for char in x_name if char.isalpha()])
    file_name = str(
        current_dir / 'picture' / f'{sampler_suffix}_' / f'17_{name[i]}_{literal_feature_name}_predict_force_plot.png')

    if Path(file_name).exists():
        Path(file_name).unlink()

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    return file_name, predict_prob


def load_app(model_index: int):
    st.set_page_config(page_title="CRAS Predictor", layout="centered")

    st.markdown(
        """
        <style>
        div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
            max-width: 500px !important;
            width: 500px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 已经彻底移除了导致报错的路径检查阻拦代码，保持原汁原味的无缝加载
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[model_index], "rb"))
    else:
        P = Load_Data()
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[model_index])
        pickle.dump(feature, open(feature_pickles[model_index], "wb"))

    index, x_name = feature.index[:], feature.x_name[:]
    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"

    top_len = top_lens[model_index]
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]

    if load_existed:
        with open(model_pickle_names[model_index], 'rb') as f:
            model = pickle.load(f)
    else:
        model_raw = copy.deepcopy(MM[model_index])
        model_raw = model_raw.fit(x_train, y_train)
        model = CalibratedClassifierCV(model_raw, method="sigmoid", cv=None, ensemble=False)
        model.fit(x_train, y_train)
        with open(model_pickle_names[model_index], "wb") as f:
            pickle.dump(model, f)

    excel_path = str(current_dir / "feature_display_info.xlsx")
    if not os.path.exists(excel_path):
        st.error(f"❌ 找不到 Excel 配置文件：{excel_path}")
        st.stop()

    data = pd.read_excel(excel_path)
    feature_dict = data.set_index('feature_name').to_dict('index')

    st.title("CRAS Mortality Predictor for ICU")

    input_features = list()
    for feature_name in x_name:
        
        lookup_name = r"\beta-blockers" if feature_name == r"$\beta$-blockers" else feature_name
        
        feature_info = feature_dict.get(lookup_name, None)
        if feature_info is None:
            continue
            
        display_type = feature_info["type"]
        display_type = display_type if isinstance(display_type, str) else ""
        
        display_name = feature_info["display_name"]
        display_name = display_name if isinstance(display_name, str) else ""
        
        display_units = feature_info["units"]
        display_units = f"units: {display_units}" if isinstance(display_units, str) else ""
        
        display_content = f"{display_name}\t{display_units} "
        
        feature_i = 0.0
        categories = []
        
        for i in range(5):
            category_i = feature_info[i]
            if isinstance(category_i, float) and np.isnan(category_i):
                continue
            categories.append(category_i)

        if display_type == "int":
            if lookup_name == "Age":
                feature_i = st.number_input(display_content, min_value=0.0, format="%.3f", key=feature_name)
            else:
                feature_i = st.number_input(display_content, min_value=0, key=feature_name)
            feature_i = float(feature_i)
            
        elif display_type == "float":
            if lookup_name == "Age":
                feature_i = st.number_input(display_content, min_value=0.0, format="%.3f", key=feature_name)
            else:
                feature_i = st.number_input(display_content, min_value=0.0, key=feature_name)
            feature_i = float(feature_i)
            
        elif display_type == "category" or display_type == "bool":
            feature_i = st.selectbox(display_content, categories, key=feature_name)
            for i, category_i in enumerate(categories):
                if feature_i != category_i:
                    continue
                feature_i = float(i)
        else:
            continue
            
        input_features.append(feature_i)

    if st.button("Execute Model Analyze"):
        with st.spinner("Calculating..."):
            force_plot_path, predict_prob = draw_force_plot(i=model_index, sample=input_features)
            prob_value = predict_prob[0]
            
            # 将最优阈值固定为 0.393
            optimal_threshold = 0.393

            st.markdown("### Prediction Result")
            st.write(f"**Predicted Mortality Probability:** {prob_value:.3f} ({(prob_value*100):.1f}%)")

            if prob_value > optimal_threshold:
                st.error(f"⚠️ **HIGH RISK** (Probability > {optimal_threshold})")
                st.markdown(f"**Clinical Implication:** The patient's mortality risk exceeds the optimal decision threshold ({optimal_threshold}). Intensive monitoring is recommended.")
            else:
                st.success(f"✅ **LOW RISK** (Probability ≤ {optimal_threshold})")
                st.markdown(f"**Clinical Implication:** The patient's mortality risk is below the optimal decision threshold.")

            st.write("---")
            st.write("**SHAP Force Plot Interpretation:**")
            st.image(force_plot_path)

if "__main__" == __name__:
    # 默认调用 RF 模型 (index 1)
    load_app(1)