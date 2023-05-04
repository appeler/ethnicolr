import streamlit as st
import pandas as pd
import ethnicolr
from ethnicolr import census_ln, pred_census_ln, pred_fl_reg_ln, pred_fl_reg_name
import base64
import matplotlib.pyplot as plt

# Define your sidebar options
sidebar_options = {
    'Append Census Data to Last Name': census_ln,
    'Florida VR Last Name Model': pred_fl_reg_ln,
    'Florida VR Full Name Model': pred_fl_reg_name
}

default_last_name_list = ["garcia, hernandez, smith, chen, washington, jackson, brown"]
default_name_list = ["john smith, john wayne, lili peng, miguel garcia, lakisha johnson"]

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def a_plot(df, group_col):
    # Group the data and count the occurrences of each group
    grouped_df = df.groupby(group_col).size().reset_index(name="Count")

    # Display the grouped data
    st.write("Grouped Data")
    st.dataframe(grouped_df)

    # Create a bar plot of the grouped data
    fig, ax = plt.subplots()
    ax.bar(grouped_df[group_col], grouped_df["Count"])
    ax.set_xlabel(group_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Counts by {group_col}")
    st.pyplot(fig)

def app():
    # Set app title
    st.title("ethnicolr: Predict Race and Ethnicity From Name")

    # Generic info.
    st.write('We use the US census data, the Florida voting registration data, \
              and the Wikipedia data collected by Skiena and colleagues, to \
              predict race and ethnicity based on first and last name or just the last name.')

    st.write("Enter a list of (last) names or upload a CSV with (first and) last name columns. Here's a [sample CSV](https://raw.githubusercontent.com/appeler/ethnicolr/master/ethnicolr/data/input-with-header.csv)")

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
            transformed_df[group_cols] = transformed_df[group_cols].astype(float)
            transformed_df["modal_race"] = transformed_df[group_cols].idxmax(axis=1).str[3:]
            st.dataframe(transformed_df)
            download_file(transformed_df)
            a_plot(transformed_df, "modal_race")
    
    elif selected_function == "Florida VR Last Name Model":
        input_type = st.radio("Input type:", ("List", "CSV"))
        if input_type == "List":
            input_list = st.text_input("Enter a list of last names (comma-separated):", value=", ".join(default_last_name_list))
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
        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col)
            st.dataframe(transformed_df)
            download_file(transformed_df)
            a_plot(transformed_df, "race")

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

                # Convert the list of dictionaries into a Pandas DataFrame
                df = pd.DataFrame(name_dicts)
                fname_col = 'fname_col'
                lname_col = 'lname_col'
        elif input_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data loaded successfully!")
                fname_col = st.selectbox("Select column with the first names", df.columns)
                lname_col = st.selectbox("Select column with the last names", df.columns)

        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col, fname_col = fname_col)
            st.dataframe(transformed_df)
            download_file(transformed_df)
            a_plot(transformed_df, group_col = 'race')

# Run the app
if __name__ == "__main__":
    app()
