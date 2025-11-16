# ml_bilstm.py
import numpy as np
import tensorflow as tf


# -----------------------------
# Create multivariate windows
# -----------------------------
def create_windows_multifeature(series, window_size):
    """
    series: shape (n_rows, n_features)
    returns:
        X -> shape (n_windows, window_size, n_features)
        y -> shape (n_windows, 1)
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i, :])
        y.append(series[i, 0])  # predict next Close
    return np.array(X), np.array(y).reshape(-1, 1)


# -----------------------------
# Build Bi-LSTM model
# -----------------------------
def build_bilstm(window_size, n_features, lstm1=64, lstm2=32, dropout=0.2):
    """
    Builds a Bidirectional LSTM model with two BiLSTM layers.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm1, return_sequences=True),
            input_shape=(window_size, n_features)
        ),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm2, return_sequences=False)
        ),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


# -----------------------------
# Train Bi-LSTM
# -----------------------------
def train_bilstm(X_train, y_train, window_size, n_features,
                 epochs=20, batch_size=32, verbose=0, model_path=None):
    """
    Trains the Bi-LSTM model with callbacks.
    Saves best model to model_path if provided.
    """

    model = build_bilstm(window_size, n_features)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3
        )
    ]

    if model_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', save_best_only=True
            )
        )

    # Use validation split only if enough data
    val_split = 0.1 if len(X_train) > 30 else 0.0

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        shuffle=True,
        verbose=verbose,
        callbacks=callbacks
    )

    # Load best model if saved
    if model_path:
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception:
            pass

    return model


# -----------------------------
# Load saved model
# -----------------------------
def load_trained_model(path: str):
    """
    Loads a trained Keras model safely.
    Returns None if not found or load fails.
    """
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None
