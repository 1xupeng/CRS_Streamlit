import numpy as np
import pandas as pd
import shap.plots

from read_toolbox import *
import streamlit as st

def prepare_model(i: int):
    if load_existed:
        P = pickle.load(open(data_pickle_name, "rb"))
        feature = pickle.load(open(feature_pickles[i], "rb"))
    else:
        P = Load_Data()  # 数据
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[i])
        pickle.dump(feature, open(feature_pickles[i], "wb"))


    # 提取特征
    index, x_name = feature.index[:], feature.x_name[:]

    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"
    # In[] 读取数据
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
        P = Load_Data()  # 数据
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[i])
        pickle.dump(feature, open(feature_pickles[i], "wb"))
    index, x_name = feature.index[:], feature.x_name[:]

    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"
    # In[] 读取数据
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
    # sample_index = 0
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
        with open(data_pickle_name, "rb") as f:
            P = pickle.load(f)
        with open(feature_pickles[model_index], "rb") as f:
            feature = pickle.load(f)
    else:
        P = Load_Data()  # 数据
        pickle.dump(P, open(data_pickle_name, "wb"))
        feature = save_read_data().read(feature_pickle_names[model_index])
        pickle.dump(feature, open(feature_pickles[model_index], "wb"))
    index, x_name = feature.index[:], feature.x_name[:]

    for i_, feature_name in enumerate(x_name):
        if feature_name == "Vasoactive":
            x_name[i_] = "Vasopressor"
        if feature_name == r"\beta-blockers":
            x_name[i_] = r"$\beta$-blockers"
    # In[] 读取数据
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

    st.title(f"CRS Prediction")
    input_features = list()
    print(feature_names)
    for feature_name in feature_names:
        print(feature_name)
        # if feature_name == "Vasopressor":
        #     feature_name = "Vasoactive"
        if feature_name == r"$\beta$-blockers":
            feature_name = r"\beta-blockers"
        # if feature_name == "ACEI_ARB":
        #     feature_name = "ACEI/ARB"

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
        if display_type == "int":
            feature_i = st.number_input(display_content, min_value=0)
            # print(f"{display_name},type:{display_type}, value:{feature_i}")
            feature_i = float(feature_i)
        elif display_type == "float":
            feature_i = st.number_input(display_content, min_value=0.0)
            # print(f"{display_name},type:{display_type}, value:{feature_i}")
            feature_i = float(feature_i)
        # elif display_type == "bool":
        #     feature_i = st.checkbox(display_content)
        #     print(f"{display_name},type:{display_type}, value:{feature_i}")
        #     feature_i = float(feature_i)
        elif display_type == "category" or display_type == "bool":
            feature_i = st.selectbox(display_content, categories)
            # print(f"{display_name},type:{display_type}, value:{feature_i}")
            for i,category_i in enumerate(categories):
                if feature_i != category_i:
                    continue
                feature_i = float(i)

        else:
            continue
        # print(feature_i)
        input_features.append(feature_i)

    # print(input_features)
    if st.button("Execute Model Analyze"):
        force_plot_path, predict_prob = draw_force_plot(i=model_index, sample=input_features)
        st.write("Result:", np.around(predict_prob, 2))
        st.image(force_plot_path)


if "__main__" == __name__:
    load_app(1)
# sample = [0,2,0,1,70.62,1]
# draw_force_plot(1,sample)
