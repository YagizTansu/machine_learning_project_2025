import pandas as pd
import numpy as np


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """Custom implementation of train-test split without using sklearn."""
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get total number of samples
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split the data
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def load_and_combine_datasets():
    """Load red and white wine datasets and combine them."""
    # Load datasets
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')
    
    # Add wine type column
    red_wine['wine_type'] = 0  # Red wine
    white_wine['wine_type'] = 1  # White wine
    
    # Combine datasets
    combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)
    

    # if quality >= 6 -> 1 else 0
    # 1 = good quality, 0 = bad quality
    combined_wine['quality'] = (combined_wine['quality'] >= 6).astype(int)

    print("******Combined wine dataset:************")
    print(combined_wine.head())
    print(combined_wine.info())
    
    print("*************Quality value counts:(    # 1 = good quality, 0 = bad quality)*********")
    print(combined_wine['quality'].value_counts())

    return combined_wine

def split_and_prepare_data(df):
    """Split the dataset into training,test sets."""
    # Split into features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split into training (80%) and test (20%) sets using custom function
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def StandardScaler(X_train, X_test):
    """Standardize features by removing the mean and scaling to unit variance."""
    mean = X_train.mean()
    std = X_train.std()
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled
