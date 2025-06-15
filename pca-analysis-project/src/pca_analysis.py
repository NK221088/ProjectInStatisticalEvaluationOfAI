import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def normalize_data(df):
    """
    Normalize the numerical data in the DataFrame.
    
    :param df: DataFrame containing the data to normalize.
    :return: Normalized DataFrame.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)

def perform_pca(df, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the given DataFrame.
    
    :param df: DataFrame containing the data to analyze.
    :param n_components: Number of principal components to return.
    :return: DataFrame containing the principal components.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    return pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)]), pca.explained_variance_ratio_

def get_pca_results(df, n_components=2):
    """
    Normalize the data and perform PCA, returning the principal components and explained variance.
    
    :param df: DataFrame containing the data to analyze.
    :param n_components: Number of principal components to return.
    :return: Tuple of (principal components DataFrame, explained variance ratio).
    """
    normalized_df = normalize_data(df)
    principal_components, explained_variance = perform_pca(normalized_df, n_components)
    return principal_components, explained_variance