import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
TICKER_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
SEQUENCE_LENGTH = 63
N_SPLITS = 5
PATIENCE = 10
MODEL_SAVE_PATH = 'stock_gru_model.keras'
SCALER_SAVE_PATH = 'stock_scalers.joblib'

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_model(input_shape):
    """Creates the Conv1D + 2-Layer Bidirectional GRU model."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Bidirectional(GRU(units=50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        Dropout(0.35),
        Bidirectional(GRU(units=50, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        Dropout(0.35),
        Dense(units=25),
        Dense(units=1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def create_sequences(features, target, seq_length):
    """Converts time series data into sequences for the model."""
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i-seq_length:i])
        y.append(target[i, 0])
    return np.array(X), np.array(y)

def predict_next_day(df_full, feature_cols, seq_len, model_path, scaler_path):
    """Predicts the stock price for the next trading day using the last sequence of data."""
    try:
        model = tf.keras.models.load_model(model_path)
        scalers = joblib.load(scaler_path)
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        last_data = df_full[feature_cols].tail(seq_len)
        
        last_date = last_data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.dayofweek > 4: # Skip weekend
            next_date += pd.Timedelta(days=1)
        
        print(f"\n--- Interactive Next-Day Prediction ---")
        print(f"Using historical data up to: {last_date.strftime('%Y-%m-%d')}")
        print(f"Predicting price for: {next_date.strftime('%Y-%m-%d')} (Approx. Next Trading Day)")
        
        scaled_data = feature_scaler.transform(last_data.values)
        X_predict = np.expand_dims(scaled_data, axis=0)
        
        predicted_scaled = model.predict(X_predict, verbose=0)
        predicted_price = target_scaler.inverse_transform(predicted_scaled)[0][0]
        
        print(f"\nPredicted Close Price for {TICKER_SYMBOL}: ${predicted_price:.2f}")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Please ensure the model and scalers were saved successfully.")


# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. Data Collection and Preprocessing ---
    print(f"Downloading stock price data for {TICKER_SYMBOL}...")
    df = yf.download(TICKER_SYMBOL, start=START_DATE, end=END_DATE)

    print(f"Downloading VIX Volatility Index data...")
    df_vix = yf.download('^VIX', start=START_DATE, end=END_DATE)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print("Data downloaded and columns flattened successfully!")

    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
    df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    df = df.join(df_vix, how='left')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # --- 2. Feature Engineering ---
    print("\nAdding technical indicators...")
    df.ta.rsi(close='Close', length=14, append=True)
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(close='Close', length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.roc(close='Close', length=20, append=True)
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Dist_from_MA200'] = df['Close'] / df['MA_200']
    df['ATR_MA'] = df['ATRr_14'].rolling(window=50).mean()
    df['ATR_Ratio'] = df['ATRr_14'] / df['ATR_MA']
    df['Is_Monday'] = (df.index.dayofweek == 0).astype(int)

    df.dropna(inplace=True)
    print(f"Data shape after feature engineering: {df.shape}")

    feature_columns = [
        'Close', 'Volume', 'VIX_Close', 'RSI_14', 'MACDh_12_26_9', 
        'BBL_20_2.0', 'BBU_20_2.0', 'ATRr_14', 'ROC_20', 
        'Dist_from_MA200', 'ATR_Ratio', 'Is_Monday'
    ]
    target_column = 'Close'

    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    print("\nUsing features:", existing_feature_columns)
    
    features = df[existing_feature_columns].values
    target = df[target_column].values.reshape(-1, 1)
    
    # --- 3. Walk-Forward Validation and Model Training ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmse_scores, mae_scores = [], []
    
    final_model = None
    final_feature_scaler = None
    final_target_scaler = None

    print(f"\nStarting Walk-Forward Validation with {N_SPLITS} splits...")

    for fold, (train_indices, test_indices) in enumerate(tscv.split(features)):
        print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")
        
        train_features, test_features = features[train_indices], features[test_indices]
        train_target, test_target = target[train_indices], target[test_indices]

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        train_features_scaled = feature_scaler.fit_transform(train_features)
        test_features_scaled = feature_scaler.transform(test_features)
        
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        train_target_scaled = target_scaler.fit_transform(train_target)
        test_target_scaled_for_metrics = target_scaler.transform(test_target)
        
        X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, SEQUENCE_LENGTH)
        X_test, y_test = create_sequences(test_features_scaled, test_target_scaled_for_metrics, SEQUENCE_LENGTH)
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
        
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train, batch_size=32, epochs=100,
            callbacks=[early_stopping], verbose=1
        )
        
        predicted_scaled = model.predict(X_test)
        predicted = target_scaler.inverse_transform(predicted_scaled)
        actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Fold {fold + 1} RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")

        if fold == N_SPLITS - 1:
            final_actual_prices = actual
            final_predicted_prices = predicted
            final_fold_dates = df.index[test_indices][SEQUENCE_LENGTH:]
            
            final_model = model
            final_feature_scaler = feature_scaler
            final_target_scaler = target_scaler
            
            print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
            final_model.save(MODEL_SAVE_PATH)
            joblib.dump({
                'feature_scaler': final_feature_scaler,
                'target_scaler': final_target_scaler,
                'feature_columns': existing_feature_columns
            }, SCALER_SAVE_PATH)
            print(f"Model and scalers saved successfully.")
            
    # --- 4. Final Results and Visualization ---
    print("\n--- Walk-Forward Validation Summary ---")
    for i, rmse in enumerate(rmse_scores):
        print(f"Fold {i+1}: RMSE = ${rmse:.2f}, MAE = ${mae_scores[i]:.2f}")

    print(f"\nAverage RMSE across all folds: ${np.mean(rmse_scores):.2f}")
    print(f"Average MAE across all folds: ${np.mean(mae_scores):.2f}")
    print("---------------------------------------")

    print("\nVisualizing predictions for the final fold...")
    plt.figure(figsize=(14, 6))
    plt.plot(final_fold_dates, final_actual_prices, color='blue', label=f'Actual {TICKER_SYMBOL} Price')
    plt.plot(final_fold_dates, final_predicted_prices, color='red', label=f'Predicted {TICKER_SYMBOL} Price')
    plt.title(f'{TICKER_SYMBOL} Stock Price Prediction (Final Fold)')
    plt.xlabel('Date')
    plt.ylabel(f'{TICKER_SYMBOL} Stock Price (USD)')
    plt.legend()
    plt.show()
    
    # --- 5. Interactive Next-Day Prediction ---
    print(f"\nRe-downloading full data for interactive prediction...")
    current_df = yf.download(TICKER_SYMBOL, start=START_DATE)
    current_vix = yf.download('^VIX', start=START_DATE)

    if isinstance(current_df.columns, pd.MultiIndex): current_df.columns = current_df.columns.get_level_values(0)
    if isinstance(current_vix.columns, pd.MultiIndex): current_vix.columns = current_vix.columns.get_level_values(0)
    current_vix = current_vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    current_df = current_df.join(current_vix, how='left')
    current_df.fillna(method='ffill', inplace=True)
    current_df.fillna(method='bfill', inplace=True)

    current_df.ta.rsi(close='Close', length=14, append=True)
    current_df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    current_df.ta.bbands(close='Close', length=20, append=True)
    current_df.ta.atr(length=14, append=True)
    current_df.ta.roc(close='Close', length=20, append=True)
    current_df['MA_200'] = current_df['Close'].rolling(window=200).mean()
    current_df['Dist_from_MA200'] = current_df['Close'] / current_df['MA_200']
    current_df['ATR_MA'] = current_df['ATRr_14'].rolling(window=50).mean()
    current_df['ATR_Ratio'] = current_df['ATRr_14'] / current_df['ATR_MA']
    current_df['Is_Monday'] = (current_df.index.dayofweek == 0).astype(int)
    current_df.dropna(inplace=True)

    predict_next_day(
        df_full=current_df, 
        feature_cols=existing_feature_columns, 
        seq_len=SEQUENCE_LENGTH, 
        model_path=MODEL_SAVE_PATH, 
        scaler_path=SCALER_SAVE_PATH
    )