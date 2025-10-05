# Naviotech-Minor-Project

# AI Stock Price Predictor

THE MODEL HAS TO BE TRAINED ON THE STOCK TO BE ABLE TO PREDICT THE NEXT TRADING DAY CLOSE

A Python script that uses a sophisticated Conv1D + Bidirectional GRU (Gated Recurrent Unit) neural network to predict future stock prices. The model is trained on historical price data and a rich set of technical and external indicators, including short-term lagged returns to improve next-day prediction accuracy.

## Description

This project fetches historical stock and volatility index (VIX) data, engineers a variety of features (including technical indicators, EMAs, stochastic oscillators, OBV, and lagged returns), and employs a robust walk-forward validation strategy to train and test the model. The final trained model and data scalers are saved, allowing for easy execution of next-day price predictions.

The model now supports **multi-stock predictions**, automatically training on any ticker provided, and plots **fold-level predictions** to visualize performance.

## Performance

The model consistently achieves strong results, with Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) values typically **well under $10** for large-cap and moderately volatile stocks.

Performance is strongest on large-cap tech stocks. Slightly higher errors are observed on highly volatile stocks like Tesla (TSLA) and Boeing (BA), which is expected due to their inherent price volatility and sensitivity to market news.

#### Stocks Tested
The model has been successfully tested on a variety of tickers, including:
- **Tech:** AAPL, GOOGL, MSFT, NVDA
- **Automotive:** TSLA
- **Aerospace:** BA
- **Finance:** JPM
- **E-commerce:** AMZN, BABA

## Features
- Fetches historical stock data and VIX data using `yfinance`.
- Enriches data with over 20 features including:
  - Technical indicators: RSI, MACD, Bollinger Bands, ATR, ROC
  - EMAs (20 & 50), Stochastic Oscillator (K/D), OBV
  - Short-term lagged returns (1, 2, 3, 5, 10-day)
  - Moving average distance and ATR ratios
  - Time-based features (e.g., Monday Effect)
- Uses a `Conv1D` layer for feature extraction and stacked `Bidirectional GRU` layers for sequence modeling.
- Employs `TimeSeriesSplit` for robust walk-forward validation, simulating real-world trading scenarios.
- Saves the final trained model and data scalers using `Keras` and `joblib`.
- Generates **fold-level prediction plots** to visualize actual vs predicted prices.
- Supports multi-stock predictions with consistent preprocessing and sequence generation.
- Includes a script to load the saved assets and predict the next trading day's price with improved accuracy due to added features.

## How to Run
1. Clone this repository.
2. Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3. Install the required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the script to train the model and generate predictions:
    ```bash
    python StockPredV2.py
    ```
5. The script will:
    - Train the model on the selected stock
    - Plot fold-level predictions
    - Save the trained model and scalers
    - Output the next trading day's predicted close
