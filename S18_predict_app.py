import numpy as np
import pandas as pd
import shap.plots
import pickle
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

# Explicit imports from func_configs as required by the S18app environment
from func_configs import (
    MM, load_existed, top_lens, name, sampler_suffix, data_pickle_name, 
    feature_pickles, feature_pickle_names, model_pickle_names, random_seed
)

from read_toolbox import *


def prepare_model(i: int):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[i], "rb"))
    else:
        P = Load_Data()  # Data
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[i])
        pickle.dump(feature, open(feature_pickles[i], "wb"))

    # Extract features
    index, x_name = feature.index[:], feature.x_name[:]

    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"

    top_len = top_lens[i]
    # read raw data
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    # apply sort features
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    # got top n features
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]
    
    if load_existed:
        with open(model_pickle_names[i], 'rb') as f:
            model = pickle.load(f)
    else:
        model = copy.deepcopy(MM[i])
        model = model.fit(x_train, y_train)
        with open(model_pickle_names[i],"wb") as f:
            pickle.dump(model, f)
    return model, x_name


def draw_force_plot(i: int, sample: list):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[i], "rb"))
    else:
        P = Load_Data()  # Data
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
    # read raw data
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    # apply sort features
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    # got top n features
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]
    
    if load_existed:
        with open(model_pickle_names[i], 'rb') as f:
            model = pickle.load(f)
    else:
        model = copy.deepcopy(MM[i])
        model = model.fit(x_train, y_train)
        with open(model_pickle_names[i],"wb") as f:
            pickle.dump(model, f)

    def model4shap(x):
        return model.predict_proba(x)[:, 1]

    raw_sample = np.array(sample).reshape(1, -1)
    sample = P.standarlize.apply_index(raw_sample, index)

    predict_prob = model.predict_proba(sample)[:, 1]
    
    plt.figure()
    model_test_explainer = shap.Explainer(model4shap, masker=x_train, feature_names=x_name, seed=P.seed)
    model_test_shap = model_test_explainer(sample)[0]
    model_test_shap.display_data = np.around(P.standarlize.reversed(sample, index), decimals=2)

    shap.plots.force(model_test_shap,
                     feature_names=x_name,
                     matplotlib=True, show=False)

    literal_feature_name = "".join([char for char in x_name if char.isalpha()])
    file_name = f'./picture/{sampler_suffix}_/17_{name[i]}_{literal_feature_name}_predict_force_plot.png'
    if Path(file_name).exists():
        Path(file_name).unlink()
    plt.savefig(file_name, dpi=300,
                bbox_inches='tight')
    plt.close()
    return file_name, predict_prob


def load_app(model_index: int):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[model_index], "rb"))
    else:
        P = Load_Data()  # Data
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
    # read raw data
    x_train, y_train, x_test, y_test, x_validate, y_validate = P.x_train, P.y_train, P.x_test, P.y_test, P.x_validate, P.y_validate
    # apply sort features
    x_train, x_test, x_validate = x_train[:, index], x_test[:, index], x_validate[:, index]
    # got top n features
    x_train, x_test, x_validate = x_train[:, :top_len], x_test[:, :top_len], x_validate[:, :top_len]
    index, x_name = index[:top_len], x_name[:top_len]
    
    if load_existed:
        with open(model_pickle_names[model_index], 'rb') as f:
            model = pickle.load(f)
    else:
        model = copy.deepcopy(MM[model_index])
        model = model.fit(x_train, y_train)
        with open(model_pickle_names[model_index],"wb") as f:
            pickle.dump(model, f)

    model_name = name[model_index]
    feature_names = x_name

    data = pd.read_excel("./feature_display_info.xlsx")
    feature_dict = data.set_index('feature_name').to_dict('index')

    st.title("CKM Mortality Predictor for ICU")

    input_features = list()
    print(feature_names)
    for feature_name in feature_names:
        print(feature_name)
        if feature_name == r"$\beta$-blockers":
            feature_name = r"\beta-blockers"

        feature_info = feature_dict.get(feature_name,None)
        if feature_info is None:
            print(f"feature {feature_name} not found")
            continue
        display_type = feature_info["type"]
        display_type = display_type if isinstance(display_type, str) else ""
        display_name = feature_info["display_name"]
        display_name = display_name if isinstance(display_name, str) else ""
        display_units = feature_info["units"]
        display_units = f"units: {display_units}"if isinstance(display_units, str) else ""
        display_content = f"{display_name}\t{display_units} "
        
        feature_i = 0.0
        categories = []
        for i in range(5):
            category_i = feature_info[i]
            if isinstance(category_i,float) and np.isnan(category_i):
                continue
            categories.append(category_i)
        
        # UPDATED INPUT LOGIC FROM S18_predict_app
        if display_type == "int":
            # Keep: If feature is Age, allow 3 decimal input
            if feature_name == "Age":
                feature_i = st.number_input(display_content, min_value=0.0, format="%.3f")
            else:
                feature_i = st.number_input(display_content, min_value=0)
            feature_i = float(feature_i)
        elif display_type == "float":
            # Keep: If feature is Age, allow 3 decimal input
            if feature_name == "Age":
                feature_i = st.number_input(display_content, min_value=0.0, format="%.3f")
            else:
                feature_i = st.number_input(display_content, min_value=0.0)
            feature_i = float(feature_i)
        elif display_type == "category" or display_type == "bool":
            feature_i = st.selectbox(display_content, categories)
            for i,category_i in enumerate(categories):
                if feature_i != category_i:
                    continue
                feature_i = float(i)
        else:
            continue
        
        input_features.append(feature_i)

    i # Preserve variable from original context
    if st.button("Execute Model Analyze"):
        force_plot_path, predict_prob = draw_force_plot(i=model_index, sample=input_features)

        # Get probability value
        prob_value = predict_prob[0]

        # -------------------------------------------------------
        # UPDATED THRESHOLD FROM S18_predict_app
        optimal_threshold = 0.5581
        # -------------------------------------------------------

        st.markdown("### Prediction Result")
        # UPDATED FORMATTING
        st.write(f"**Predicted Mortality Probability:** {prob_value:.3f} ({(prob_value*100):.1f}%)")

        # Logic for High/Low Risk
        if prob_value > optimal_threshold:
            # High Risk (Red)
            st.error(f"⚠️ **HIGH RISK** (Probability > {optimal_threshold})")
            st.markdown(f"**Clinical Implication:** The patient's mortality risk exceeds the optimal decision threshold ({optimal_threshold}). Intensive monitoring is recommended.")
        else:
            # Low Risk (Green)
            st.success(f"✅ **LOW RISK** (Probability ≤ {optimal_threshold})")
            st.markdown(f"**Clinical Implication:** The patient's mortality risk is below the optimal decision threshold.")

        st.write("---")
        st.write("**SHAP Force Plot Interpretation:**")
        st.image(force_plot_path)


if "__main__" == __name__:
    load_app(1)