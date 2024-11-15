import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from openai import OpenAI

# Initialize OpenAI client using Streamlit's secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Title of the app
st.title("Advanced EDA Bot - Data Science Assistant")

# Brief description
st.write("Welcome to the Advanced EDA Bot. Upload a CSV file, and I will help you perform Exploratory Data Analysis (EDA) and visualize your data.")

# Accept CSV file upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Function to detect delimiter using a fallback mechanism
def detect_delimiter(file):
    # Try common delimiters
    possible_delimiters = [',', '\t', ';', ' ']
    
    # Try reading the file with each delimiter
    for delimiter in possible_delimiters:
        try:
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file, delimiter=delimiter)
            # Check if dataframe has more than one column to confirm successful read
            if df.shape[1] > 1:
                return delimiter
        except Exception as e:
            continue
    # If no delimiter works, raise an error
    raise ValueError("Could not determine the delimiter. Please check the file format.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get a response from OpenAI with EDA-specific context
def get_response(prompt, df=None):
    system_message = {
        "role": "system",
        "content": ("You are an advanced data science assistant with expertise in performing Exploratory Data Analysis (EDA). "
                    "You help users analyze datasets, summarize the data, generate visualizations (like histograms, scatter plots, etc.), "
                    "and assist with data cleaning, missing value handling, and more. You can process data, offer suggestions, and provide insights.")
    }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] +
        [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Process and display response if there's input
if uploaded_file:
    # Detect the delimiter
    try:
        delimiter = detect_delimiter(uploaded_file)
        # Reset file pointer after reading for delimiter detection
        uploaded_file.seek(0)
        
        # Read the uploaded CSV file into a Pandas DataFrame using the detected delimiter
        df = pd.read_csv(uploaded_file, delimiter=delimiter)
        
    except Exception as e:
        st.error(f"Error detecting delimiter: {e}")
        st.stop()
    
    # Show dataset overview and summary statistics
    st.subheader("Dataset Overview")
    st.write(df.head())  # Show first few rows of the dataset
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Ask the assistant for an EDA task
    user_input = st.chat_input("What EDA task would you like to perform? (e.g., summary statistics, correlation, visualize distribution, etc.)")

    if user_input:
        # Append user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response from GPT assistant (for EDA tasks)
        assistant_response = get_response(user_input, df)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    # Perform basic EDA
    st.subheader("Basic Exploratory Data Analysis")

    # Summary statistics
    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe(include="all"))  # Include all columns for summary

    # Show data types
    st.subheader("Data Types of Columns")
    st.write(df.dtypes)

    # Check for missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation Heatmap (only for numeric columns)
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for correlation.")

    # Visualize distributions of numeric columns
    st.subheader("Distribution of Numeric Columns")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=30, color='blue', alpha=0.7)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Pairplot for visualizing relationships (only if dataset is small)
    if len(df) <= 1000:  # Limiting the size for performance reasons
        st.subheader("Pairplot of the Dataset")
        sns.pairplot(df)
        st.pyplot(plt)

    # Show boxplots for numeric columns
    st.subheader("Boxplots for Numeric Columns")
    for column in numeric_columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax, color='orange')
        ax.set_title(f'Boxplot of {column}')
        st.pyplot(fig)

    # Data Cleaning options
    st.subheader("Data Cleaning Options")

    # Drop rows with missing values option
    if st.checkbox("Drop rows with missing values"):
        df_cleaned = df.dropna()
        st.write(f"Cleaned Data Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
        st.write(df_cleaned.head())
    
    # Fill missing values with median or mean for numeric columns
    if st.checkbox("Fill missing numeric values with median"):
        df_filled = df.copy()  # Make a copy to avoid modifying the original df
        numeric_columns = df_filled.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            df_filled[column].fillna(df_filled[column].median(), inplace=True)
        st.write(f"Filled Data Shape: {df_filled.shape[0]} rows, {df_filled.shape[1]} columns")
        st.write(df_filled.head())

    # Provide the option to download the cleaned data
    if st.button("Download Cleaned Data"):
        cleaned_csv = df_filled.to_csv(index=False)
        st.download_button(label="Download CSV", data=cleaned_csv, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.write("Please upload a CSV file to begin the EDA.")
