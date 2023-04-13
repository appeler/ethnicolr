import streamlit as st
import pandas as pd
import ethnicolr
from ethnicolr import census_ln

# Set app title
st.title("Ethnicolr")

# Add a file uploader widget for the user to upload a CSV file

def app():
    # Set app title
    st.title("My Streamlit App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
    else:
        st.stop()

    lname_col = st.selectbox("Select column with last name", df.columns)
    year = st.selectbox("Select a year", [2000, 2010])

    if st.button("Append Electoral Roll Data"):
    # Use the package to transform the DataFrame
        transformed_df = census_ln(df, lname_col=lname_col)
        st.dataframe(transformed_df)

        csv = transformed_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
        st.markdown(href, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    app()
