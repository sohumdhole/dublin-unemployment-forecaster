import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# Suppress tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_arima(series, steps=4):
    """
    Train an ARIMA model and forecast the next 'steps' periods.
    """
    # Simply using a common parametrization: order=(5,1,0)
    # We can tune this in a more robust app, but this provides a strong baseline.
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=steps)
    
    # Also get in-sample predictions for performance metrics
    in_sample = model_fit.predict(start=0, end=len(series)-1)
    rmse = np.sqrt(mean_squared_error(series, in_sample))
    
    return forecast, in_sample, rmse

def build_prophet(df, date_col, value_col, steps=4):
    """
    Train a Prophet model
    """
    # Prophet requires 'ds' and 'y' columns
    p_df = pd.DataFrame({
        'ds': df[date_col],
        'y': df[value_col]
    })
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(p_df)
    
    # Make future dataframe format (Quarterly)
    future = model.make_future_dataframe(periods=steps, freq='QS')
    forecast = model.predict(future)
    
    # Extract just the forecast
    future_pred = forecast.iloc[-steps:]['yhat'].values
    
    # In-sample
    in_sample = forecast.iloc[:-steps]['yhat'].values
    rmse = np.sqrt(mean_squared_error(p_df['y'], in_sample))
    
    # Create corresponding dates for the forecast
    future_dates = forecast.iloc[-steps:]['ds'].tolist()
    
    return future_pred, in_sample, rmse, future_dates

def build_lstm(series, steps=4, epochs=50):
    """
    Train an LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    # Prepare sequence data
    seq_size = 4  # Use past 4 quarters to predict the next
    X, y = [], []
    for i in range(len(scaled_data) - seq_size):
        X.append(scaled_data[i:(i + seq_size), 0])
        y.append(scaled_data[i + seq_size, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=4, verbose=0)
    
    # In-sample predictions
    # We pad the beginning since we need sequence length to predict
    train_predict = model.predict(X, verbose=0)
    train_predict_inv = scaler.inverse_transform(train_predict).flatten()
    
    # Backfill first 'seq_size' elements with actuals for visual comparison
    in_sample = np.concatenate([series.values[:seq_size], train_predict_inv])
    rmse = np.sqrt(mean_squared_error(series.values, in_sample))
    
    # Predict future
    last_seq = scaled_data[-seq_size:]
    future_preds = []
    
    current_seq = last_seq
    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, seq_size, 1), verbose=0)[0]
        future_preds.append(pred)
        # Shift sequence
        current_seq = np.append(current_seq[1:], pred)
        
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    
    return future_preds_inv, in_sample, rmse

