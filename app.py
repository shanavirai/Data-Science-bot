import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Advanced EDA Tool for Data Science Professionals")

# Brief description of the tool
st.write("""
This app allows you to upload a CSV file and perform advanced Exploratory Data Analysis (EDA).
You can explore various aspects of the dataset, including summaries, visualizations, and more!
""")

# Accept CSV file upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.subheader("Dataset Overview")
    st.write(df.head())  # Show first few rows of the dataset
    
    # Show dataset dimensions
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Show column types
    st.subheader("Data Types of Columns")
    st.write(df.dtypes)

    # Show missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Show correlations
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(plt)

    # Show distribution of numeric columns
    st.subheader("Distribution of Numeric Columns")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=30, color='blue', alpha=0.7)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Pairplot (if the dataset is small)
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

    # Correlation matrix (pairwise) for specific features
    feature_select = st.multiselect("Select features to analyze correlation", df.columns.tolist())
    if feature_select:
        st.subheader("Correlation of Selected Features")
        selected_corr = df[feature_select].corr()
        st.write(selected_corr)

    # Pairwise plot for selected features
    if len(feature_select) > 1:
        st.subheader("Pairplot of Selected Features")
        sns.pairplot(df[feature_select])
        st.pyplot(plt)

    # Data Cleaning options
    st.subheader("Data Cleaning Options")

    # Drop rows with missing values option
    if st.checkbox("Drop rows with missing values"):
        df_cleaned = df.dropna()
        st.write(f"Cleaned Data Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
        st.write(df_cleaned.head())
    
    # Fill missing values with median or mean for numeric columns
    if st.checkbox("Fill missing numeric values with median"):
        df_filled = df.fillna(df.median())
        st.write(f"Filled Data Shape: {df_filled.shape[0]} rows, {df_filled.shape[1]} columns")
        st.write(df_filled.head())

    # Provide the option to download the cleaned data
    if st.button("Download Cleaned Data"):
        cleaned_csv = df_cleaned.to_csv(index=False)
        st.download_button(label="Download CSV", data=cleaned_csv, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.write("Please upload a CSV file to begin the EDA.")

