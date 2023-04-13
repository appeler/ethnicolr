import streamlit as st
import pandas as pd
import ethnicolr
from ethnicolr import census_ln

# Set app title
st.title("Ethnicolr")

# Add a file uploader widget for the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

lname_col = st.selectbox("Select column with last name", df.columns)
year = st.selectbox("Select a year", [2000, 2010])

# Add a button to trigger the transformation
if st.button("Append Electoral Roll Data"):
    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the uploaded CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Use the package to transform the DataFrame
        transformed_df = census_ln(df, col=lname_col)
        
        # Display the transformed DataFrame as a table
        st.dataframe(transformed_df)

        # Download results as CSV
        csv = transformed_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        # Display an error message if no file was uploaded
        st.error("Please upload a CSV file to transform.")
