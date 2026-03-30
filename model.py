import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(df):
    df_scaled, scaler, pca = preprocess_data(df)

    # Split into training and test sets
    train_size = int(len(df_scaled) * 0.8)
    train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

    # Create sequences
    SEQ_LENGTH = 10
    X_train, y_train = create_sequences(train_data.values, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_data.values, SEQ_LENGTH)

    # Define LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(y_train.shape[1])
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    return model, scaler, pca


def predict_future(model, df, scaler, pca, future_steps=12):
    df_scaled, _, _ = preprocess_data(df)

    last_seq = df_scaled.values[-10:]  # Last sequence for prediction
    future_preds = []

    for _ in range(future_steps):
        next_pred = model.predict(last_seq.reshape(1, 10, -1))
        future_preds.append(next_pred.flatten())
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1] = next_pred

    return pca.inverse_transform(future_preds)
