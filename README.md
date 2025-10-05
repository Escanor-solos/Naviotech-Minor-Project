# Naviotech-Minor-Project
<br>
# AI Stock Price Predictor

A Python script that uses a sophisticated Conv1D + Bidirectional GRU (Gated Recurrent Unit) neural network to predict future stock prices. The model is trained on historical price data and a rich set of technical and external indicators.

## Description

This project fetches historical stock and volatility index (VIX) data, engineers a variety of features, and employs a robust walk-forward validation strategy to train and test the model. The final trained model and data scalers are saved, allowing for easy execution of next-day price predictions.

The model's performance has been validated across a diverse set of stocks from different sectors, demonstrating its versatility.

## Performance

The model consistently achieves strong results, with Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) values typically **well under $10**.

Performance is strongest on large-cap tech stocks. Higher error values were observed on more volatile stocks like Tesla (TSLA) and Boeing (BA), which is expected due to their inherent price volatility and sensitivity to market news.

#### Stocks Tested
The model has been successfully tested on a variety of tickers, including:
- **Tech:** AAPL, GOOGL, MSFT
- **Automotive:** TSLA
- **Aerospace:** BA
- **Finance:** JPM

## Features
- Fetches historical stock data and VIX data using `yfinance`.
- Enriches data with over 10 technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.).
- Incorporates external factors like the VIX and time-based features (e.g., the "Monday Effect").
- Uses a `Conv1D` layer for feature extraction and stacked `Bidirectional GRU` layers for sequence modeling.
- Employs `TimeSeriesSplit` for robust walk-forward validation, simulating real-world trading scenarios.
- Saves the final trained model and data scalers using `Keras` and `joblib`.
- Includes a script to load the saved assets and predict the next trading day's price.

## How to Run
1.  Clone this repository.
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Install the required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the script to train the model and generate a prediction:
    ```bash
    python StockPredV2.py
    ```
