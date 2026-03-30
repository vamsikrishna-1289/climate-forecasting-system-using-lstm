import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def preprocess_data(df):
    df.fillna(method="ffill", inplace=True)

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=["Location", "Country"])

    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # Reduce dimensions using PCA
    pca = PCA(n_components=50)  # Adjust as needed
    df_reduced = pca.fit_transform(df_scaled)

    return pd.DataFrame(df_reduced, index=df_scaled.index), scaler, pca
