import streamlit as st
import pandas as pd
import ethnicolr
from ethnicolr import census_ln, pred_census_ln, pred_fl_reg_ln, pred_fl_reg_name
import base64


# Define your sidebar options
sidebar_options = {
    'Append Census Data to Last Name': census_ln,
    'Florida VR Last Name Model': pred_fl_reg_ln,
    'Florida VR Full Name Model': pred_fl_reg_name
}

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("ethnicolr: Predict Race and Ethnicity From Name")

    # Generic info.
    st.write('We use the US census data, the Florida voting registration data, \
              and the Wikipedia data collected by Skiena and colleagues, to \
              predict race and ethnicity based on first and last name or just the last name.')
    st.write('[Github Repository](https://github.com/appeler/ethnicolr)')

    # Set up the sidebar
    st.sidebar.title('Select Function')
    selected_function = st.sidebar.selectbox('', list(sidebar_options.keys()))

    if selected_function == "Append Census Data to Last Name":
        input_type = st.radio("Input type:", ("List", "CSV"))
        if input_type == "List":
            input_list = st.text_input("Enter a list of last names (comma-separated)")
            year = st.selectbox("Select a year", [2000, 2010])
            if input_list:
                input_list = input_list.split(",")
                list_name = [x.strip() for x in input_list]
                df = pd.DataFrame(list_name, columns=['lname_col'])
                lname_col = 'lname_col'
        elif input_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            # Load data
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data loaded successfully!")
                lname_col = st.selectbox("Select column with last name", df.columns)
                year = st.selectbox("Select a year", [2000, 2010])
        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, lname_col=lname_col, year = year)
            st.dataframe(transformed_df)
            download_file(transformed_df)
    
    elif selected_function == "Florida VR Last Name Model":
        lname_col = st.selectbox("Select column with last name", df.columns)
        function = sidebar_options[selected_function]
        if st.button('Run'):
            transformed_df = function(df, namecol=lname_col)
            st.dataframe(transformed_df)
            download_file(transformed_df)

    


# Run the app
if __name__ == "__main__":
    app()
