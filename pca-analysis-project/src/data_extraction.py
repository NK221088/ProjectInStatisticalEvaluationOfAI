import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the processed data from a CSV file into a DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def preprocess_data(df):
    """
    Preprocess the DataFrame by normalizing the numerical columns.
    
    :param df: DataFrame to preprocess.
    :return: Normalized DataFrame.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def extract_groups(df, groups):
    """
    Extract specified groups from the DataFrame.
    
    :param df: DataFrame containing the data.
    :param groups: Dictionary of groups to extract.
    :return: DataFrame with only the specified groups.
    """
    return df[list(groups.keys())]

def clean_column_names(df):
    """
    Clean the column names of the DataFrame by removing spaces and special characters.
    
    :param df: DataFrame to clean.
    :return: DataFrame with cleaned column names.
    """
    df.columns = [re.sub(r'\W+', '_', col).strip() for col in df.columns]
    return df

def prepare_data(file_path, groups):
    """
    Load, preprocess, and extract groups from the data.
    
    :param file_path: Path to the CSV file.
    :param groups: Dictionary of groups to extract.
    :return: DataFrame ready for PCA analysis.
    """
    df = load_data(file_path)
    df = clean_column_names(df)
    df = preprocess_data(df)
    return extract_groups(df, groups)