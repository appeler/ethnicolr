import streamlit as st
import pandas as pd
import ethnicolr
import json
import datetime
from ethnicolr import census_ln, pred_census_ln, pred_fl_reg_ln, pred_fl_reg_name
import base64

sidebar_options = {
    'Append Census Data to Last Name': census_ln,
    'Florida VR Last Name Model': pred_fl_reg_ln,
    'Florida VR Full Name Model': pred_fl_reg_name
}

default_last_name_list = ["garcia, hernandez, smith, chen, washington, jackson, brown"]
default_name_list = ["john smith, john wayne, lili peng, miguel garcia, lakisha johnson"]

# Load the usage logs from a JSON file
try:
    with open("usage_logs.json", "r") as f:
        usage_logs = json.load(f)
except FileNotFoundError:
    usage_logs = {}

def log_usage(action):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    if action in usage_logs:
        usage_logs[action].append(date_str)
    else:
        usage_logs[action] = [date_str]
    with open("usage_logs.json", "w") as f:
        json.dump(usage_logs, f)
    global count
    count = len(usage_logs.get("app", []))

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():

    log_usage("ethnicolr")

    st.markdown(
    """
    ### Predict Race and Ethnicity From Name

    We use the US census data and the Florida voting registration data to\
    predict race and ethnicity based on first and last name or just the last name.'

    #### Aim
    
    We are releasing this software in the hope that it enables activists and researchers\
    
    1. Highlight biases
    2. Fight biases
    3. Prevent biases

    Here are a couple of papers: 1. [Diversity of news recommendations](https://dl.acm.org/doi/abs/10.1145/3406522.3446019) 2. [Diversity Innovation Paradox](https://www.pnas.org/doi/abs/10.1073/pnas.1915378117) that have used our software

    #### Usage

    Enter a list of (last) names or upload a CSV with (first and) last name columns. Here's a [sample CSV](https://raw.githubusercontent.com/appeler/ethnicolr/master/ethnicolr/data/input-with-header.csv)

    """
    )

    st.write(f"Current usage count: {count}")

    int_range = (0, 100000)
    float_range = (0.0, 1.0)

    # Set up the sidebar
    selected_function = st.sidebar.selectbox(label = 'Select a function', options = list(sidebar_options.keys()))

    if selected_function == "Append Census Data to Last Name":
        input_type = st.radio("Input type:", ("List", "CSV"))
        if input_type == "List":
            input_list = st.text_input("Enter a list of last names (comma-separated):", value=", ".join(default_last_name_list))
            year = st.selectbox(label = "Select a year", options = [2000, 2010])
            if input_list:
                input_list = input_list.split(",")
                list_name = [x.strip() for x in input_list]
                df = pd.DataFrame(list_name, columns=['lname_col'])
                lname_col = 'lname_col'
        elif input_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data loaded successfully!")
                lname_col = st.selectbox("Select column with last name", df.columns)
                year = st.selectbox(label = "Select a year", options = [2000, 2010])
        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col, year = year)
            group_cols = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
            st.dataframe(transformed_df)
            download_file(transformed_df)
    
    elif selected_function == "Florida VR Last Name Model":
        input_type = st.radio("Input type:", ("List", "CSV"))
        if input_type == "List":
            input_list = st.text_input("Enter a list of last names (comma-separated):", value=", ".join(default_last_name_list))
            if input_list:
                input_list = input_list.split(",")
                list_name = [x.strip() for x in input_list]
                df = pd.DataFrame(list_name, columns=['lname_col'])
                lname_col = 'lname_col'
                iter_val = st.sidebar.number_input("Enter number of iterations", min_value=int_range[0], max_value=int_range[1], step=1)
                conf_int_val = st.sidebar.number_input("Enter confidence interval (0 to 1)", min_value=float_range[0], max_value=float_range[1], step=0.01)

        elif input_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data loaded successfully!")
                lname_col = st.selectbox("Select column with last name", df.columns)
                iter_val = st.sidebar.number_input("Enter number of iterations", min_value=int_range[0], max_value=int_range[1], step=1)
                conf_int_val = st.sidebar.number_input("Enter confidence interval (0 to 1)", min_value=float_range[0], max_value=float_range[1], step=0.01)

        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col, conf_int = conf_int_val, num_iter = iter_val)
            st.dataframe(transformed_df)
            download_file(transformed_df)

    elif selected_function == "Florida VR Full Name Model":
        input_type = st.radio("Input type:", ("List", "CSV"))
        if input_type == "List":
            input_list = st.text_input("Enter a list of names (comma-separated):", value=", ".join(default_name_list))
            if input_list:
                input_list = input_list.split(",")
                list_name = [x.strip() for x in input_list]
                name_dicts = []
                for name in list_name:
                    first_name, last_name = name.split(" ")
                    name_dicts.append({'fname_col': first_name, "lname_col": last_name})

                df = pd.DataFrame(name_dicts)
                fname_col = 'fname_col'
                lname_col = 'lname_col'
                iter_val = st.sidebar.number_input("Enter number of iterations", min_value=int_range[0], max_value=int_range[1], step=1)
                conf_int_val = st.sidebar.number_input("Enter confidence interval (0 to 1)", min_value=float_range[0], max_value=float_range[1], step=0.01)

        elif input_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data loaded successfully!")
                fname_col = st.selectbox("Select column with the first names", df.columns)
                lname_col = st.selectbox("Select column with the last names", df.columns)
                iter_val = st.sidebar.number_input("Enter number of iterations", min_value=int_range[0], max_value=int_range[1], step=1)
                conf_int_val = st.sidebar.number_input("Enter confidence interval (0 to 1)", min_value=float_range[0], max_value=float_range[1], step=0.01)

        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col, fname_col = fname_col, conf_int = conf_int_val, num_iter = iter_val)
            st.dataframe(transformed_df)
            download_file(transformed_df)

# Run the app
if __name__ == "__main__":
    app()
