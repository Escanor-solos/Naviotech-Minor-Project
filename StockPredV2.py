import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# ============================================================================== 
# CONFIGURATION PARAMETERS
# ============================================================================== 
SEQUENCE_LENGTH = 50
N_SPLITS = 5
PATIENCE = 10
MODEL_SAVE_PATH = 'stock_gru_model.keras'
SCALER_SAVE_PATH = 'stock_scalers.joblib'
SCALER_WINDOW = 504

# ============================================================================== 
# HELPER FUNCTIONS
# ============================================================================== 
def create_model(input_shape):
    """Creates the Conv1D + 2-Layer Bidirectional GRU model."""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        Dropout(0.25),
        Bidirectional(GRU(50, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        Dropout(0.25),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='mean_squared_error')
    return model

def create_sequences(features, target, seq_len):
    """Converts time series data into sequences for the model."""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(target[i, 0])
    return np.array(X), np.array(y)

def prepare_features(df):
    """Adds technical indicators, returns, and other features."""
    df.ta.rsi(close='Close', length=14, append=True)
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(close='Close', length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.roc(close='Close', length=20, append=True)
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Dist_from_MA200'] = df['Close'] / df['MA_200']
    df['ATR_MA'] = df['ATRr_14'].rolling(50).mean()
    df['ATR_Ratio'] = df['ATRr_14'] / df['ATR_MA']
    df['Is_Monday'] = (df.index.dayofweek == 0).astype(int)

    # Short-term lagged returns
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_2'] = df['Close'].pct_change(2)
    df['Return_3'] = df['Close'].pct_change(3)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Stochastic Oscillator
    if all(col in df.columns for col in ['High','Low','Close']):
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        for col in ['STOCHk_14_3','STOCHd_14_3']:
            if col in stoch.columns:
                df[col] = stoch[col]

    # On-Balance Volume
    if 'Close' in df.columns and 'Volume' in df.columns:
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def predict_next_day(df_full, feature_cols, seq_len, model_path, scaler_window=SCALER_WINDOW):
    """Predicts the stock price for the next trading day using rolling scaler."""
    model = tf.keras.models.load_model(model_path)
    recent_data = df_full[feature_cols].tail(scaler_window)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(recent_data.values)
    target_scaler = MinMaxScaler()
    target_scaler.fit(df_full['Close'].tail(scaler_window).values.reshape(-1,1))
    last_data = df_full[feature_cols].tail(seq_len)
    scaled_data = feature_scaler.transform(last_data.values)
    X_pred = np.expand_dims(scaled_data, axis=0)
    pred_scaled = model.predict(X_pred, verbose=0)
    predicted_price = target_scaler.inverse_transform(pred_scaled)[0][0]
    return predicted_price

# ============================================================================== 
# MAIN EXECUTION SCRIPT
# ============================================================================== 
if __name__ == "__main__":
    stock_list = ["AMD"]
    for TICKER in stock_list:
        print(f"\nDownloading stock price data for {TICKER}...")
        df = yf.download(TICKER, start="2015-01-01")
        df_vix = yf.download('^VIX', start="2015-01-01")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(df_vix.columns, pd.MultiIndex):
            df_vix.columns = df_vix.columns.get_level_values(0)

        df_vix = df_vix[['Close']].rename(columns={'Close':'VIX_Close'})
        df = df.join(df_vix, how='left')
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        required_cols = ['Close','High','Low','Volume']
        if all(col in df.columns for col in required_cols):
            df = prepare_features(df)
        else:
            print(f"Skipping {TICKER}, missing required columns for indicators.")
            continue

        feature_cols = ['Close','Volume','VIX_Close','RSI_14','MACDh_12_26_9',
                        'BBL_20_2.0','BBU_20_2.0','ATRr_14','ROC_20',
                        'Dist_from_MA200','ATR_Ratio','Is_Monday',
                        'Return_1','Return_2','Return_3','Return_5','Return_10',
                        'EMA_20','EMA_50','STOCHk_14_3','STOCHd_14_3','OBV']
        feature_cols = [col for col in feature_cols if col in df.columns]

        features = df[feature_cols].values
        target = df['Close'].values.reshape(-1,1)

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        rmse_scores, mae_scores = [], []
        final_model = None
        final_feature_scaler = None
        final_target_scaler = None

        print(f"\nStarting Walk-Forward Validation for {TICKER} with {N_SPLITS} splits...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")
            train_X, test_X = features[train_idx], features[test_idx]
            train_y, test_y = target[train_idx], target[test_idx]

            feature_scaler = MinMaxScaler()
            train_X_scaled = feature_scaler.fit_transform(train_X)
            test_X_scaled = feature_scaler.transform(test_X)

            target_scaler = MinMaxScaler()
            train_y_scaled = target_scaler.fit_transform(train_y)
            test_y_scaled = target_scaler.transform(test_y)

            X_train_seq, y_train_seq = create_sequences(train_X_scaled, train_y_scaled, SEQUENCE_LENGTH)
            X_test_seq, y_test_seq = create_sequences(test_X_scaled, test_y_scaled, SEQUENCE_LENGTH)

            print(f"Training on {len(X_train_seq)} samples, testing on {len(X_test_seq)} samples.")
            model = create_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            early_stop = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
            model.fit(X_train_seq, y_train_seq, batch_size=32, epochs=100, callbacks=[early_stop], verbose=1)

            pred_scaled = model.predict(X_test_seq)
            pred = target_scaler.inverse_transform(pred_scaled)
            actual = target_scaler.inverse_transform(y_test_seq.reshape(-1,1))

            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            print(f"Fold {fold+1} RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")

            if fold == N_SPLITS-1:
                final_model = model
                final_feature_scaler = feature_scaler
                final_target_scaler = target_scaler
                final_actual_prices = actual
                final_predicted_prices = pred
                final_fold_dates = df.index[test_idx][SEQUENCE_LENGTH:]

        # --- Plot final fold predictions ---
        print(f"\nVisualizing predictions for the final fold of {TICKER}...")
        plt.figure(figsize=(14,6))
        plt.plot(final_fold_dates, final_actual_prices, color='blue', label='Actual')
        plt.plot(final_fold_dates, final_predicted_prices, color='red', label='Predicted')
        plt.title(f'{TICKER} Stock Price Prediction (Final Fold)')
        plt.xlabel('Date')
        plt.ylabel(f'{TICKER} Stock Price (USD)')
        plt.legend()
        plt.show()

        # --- Save final model and scalers ---
        print(f"\nSaving final model and scalers for {TICKER}...")
        final_model.save(MODEL_SAVE_PATH)
        joblib.dump({
            'feature_scaler': final_feature_scaler,
            'target_scaler': final_target_scaler,
            'feature_columns': feature_cols
        }, SCALER_SAVE_PATH)
        print("Saved successfully.")

        # --- Next-Day Prediction ---
        next_price = predict_next_day(df, feature_cols, SEQUENCE_LENGTH, MODEL_SAVE_PATH)
        print(f"\nPredicted Next-Day Close for {TICKER}: ${next_price:.2f}")
