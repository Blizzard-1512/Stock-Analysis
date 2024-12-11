import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

# Custom CSS to enhance the app's appearance
# Set page config
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to enhance the app's appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #d1c9c9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        color: #333;
        margin-bottom: 10px;
        font-size: 18px;
    }
    .metric-card .value {
        font-size: 24px;
        font-weight: bold;
        color: #1e88e5;
    }
    .prediction-table {
        background-color: #0e1117;
        border-radius: 10px;
        margin: 20px 0;
        color: white; /* Default white text for visibility */
    }
    .prediction-table .stDataFrame {
        background-color: #0e1117 !important;
        color: white !important;
    }
    .prediction-table .stDataFrame thead {
        background-color: #0e1117 !important;
        color: white !important;
    }
    .prediction-table .stDataFrame th {
        background-color: #0e1117 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

class StockPredictor:
    def __init__(self, ticker: str, years: int = 10):
        self.ticker = ticker.upper()
        self.years = years
        self.data = None
        self.model_fit = None
        self.predictions = None
        self.metrics = {}
        self.var_metrics = {}
        self.models = {
            'TAES': None,  # Trend-adjusted Exponential Smoothing (existing method)
            'LSTM': None,  # Placeholder for LSTM model
            'RNN': None,   # Placeholder for RNN model
            'ARIMA': None  # Placeholder for ARIMA model
        }
        self.model_predictions = {}

    def fetch_data(self) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)

        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'),
                                 end=end_date.strftime('%Y-%m-%d'))

            if data.empty or len(data) < 252:  # Minimum one year of trading days
                raise ValueError(f"Insufficient data for {self.ticker}")

            data.index = pd.to_datetime(data.index)

            # Feature engineering
            data['Returns'] = data['Close'].pct_change()
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['Upper_Band'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
            data['Lower_Band'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)

            data = data.fillna(method='ffill')

            self.data = data
            return data

        except Exception as e:
            raise ValueError(f"Error fetching data for {self.ticker}: {str(e)}")

    def prepare_ml_data(self, test_size=0.2, look_back=60):
        """
        Prepare data for machine learning models
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        # Use closing prices for prediction
        prices = self.data['Close'].values
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

        # Create sequences for time series prediction
        X, y = [], []
        for i in range(len(scaled_prices) - look_back):
            X.append(scaled_prices[i:i+look_back])
            y.append(scaled_prices[i+look_back])
        
        X, y = np.array(X), np.array(y)

        # Split into train and test sets
        split = int(len(X) * (1-test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'look_back': look_back
        }

    def train_taes_model(self, validation_size: int = 30):
        """
        Train Trend-Adjusted Exponential Smoothing (existing method)
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
            
        prices = self.data['Close']
        self.last_train_price = prices.iloc[-1]
        daily_changes = prices.pct_change().dropna()
        self.avg_daily_change = daily_changes.mean()
        
        last_validation_prices = prices.tail(validation_size)
        x = np.arange(len(last_validation_prices))
        slope, _ = np.polyfit(x, last_validation_prices.values, 1)
        self.trend = slope / self.last_train_price
        
        metrics = {
            'Last Training Price': self.last_train_price,
            'Average Daily Change': self.avg_daily_change,
            'Trend': self.trend,
            'Method': 'Term Adjusted Exponential Smoothing'
        }
        
        self.models['TAES'] = {
        'metrics': metrics,
        'last_train_price': self.last_train_price,
        'avg_daily_change': self.avg_daily_change,
        'trend': self.trend
        }
        return metrics
        
    def train_lstm_model(self):
        """
        Train LSTM model (placeholder implementation)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            # Prepare data
            data = self.prepare_ml_data()
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            scaler = data['scaler']

            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
                LSTM(50, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # Train the model
            history = model.fit(X_train, y_train, 
                                epochs=20, 
                                batch_size=32, 
                                validation_data=(X_test, y_test),
                                verbose=0)

            # Make predictions
            predictions = model.predict(X_test)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(predictions)
            y_test_orig = scaler.inverse_transform(y_test)

            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test_orig, predictions)
            rmse = np.sqrt(mean_squared_error(y_test_orig, predictions))

            # Store model and metrics
            self.models['LSTM'] = {
                'model': model,
                'metrics': {
                    'MAPE': mape,
                    'RMSE': rmse,
                    'Method': 'LSTM'
                },
                'scaler': scaler
            }

            return self.models['LSTM']['metrics']

        except ImportError:
            st.warning("TensorFlow/Keras not available. Skipping LSTM model.")
            return None

    def train_rnn_model(self):
        """
        Train RNN model (placeholder implementation)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import SimpleRNN, Dense
            
            # Prepare data
            data = self.prepare_ml_data()
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            scaler = data['scaler']

            # Build RNN model
            model = Sequential([
                SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # Train the model
            history = model.fit(X_train, y_train, 
                                epochs=20, 
                                batch_size=32, 
                                validation_data=(X_test, y_test),
                                verbose=0)

            # Make predictions
            predictions = model.predict(X_test)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(predictions)
            y_test_orig = scaler.inverse_transform(y_test)

            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test_orig, predictions)
            rmse = np.sqrt(mean_squared_error(y_test_orig, predictions))

            # Store model and metrics
            self.models['RNN'] = {
                'model': model,
                'metrics': {
                    'MAPE': mape,
                    'RMSE': rmse,
                    'Method': 'RNN'
                },
                'scaler': scaler
            }

            return predictions

        except ImportError:
            st.warning("TensorFlow/Keras not available. Skipping RNN model.")
            return None

    def train_arima_model(self):
        """
        Train ARIMA model (placeholder implementation)
        """
        try:
            # Prepare data (using closing prices)
            prices = self.data['Close'].dropna().values
            if len(prices) == 0:
                raise ValueError("Insufficient data available for ARIMA model training.")
            
            # Fit ARIMA model
            # Note: In a real implementation, you'd use grid search or auto_arima 
            # to find the best parameters
            model = ARIMA(prices, order=(5,1,0)).fit()

            # Make in-sample predictions
            predictions = model.predict()
            
            if len(predictions) == 0:
                raise ValueError("ARIMA predictions failed due to insufficient data.")

            predictions = np.array(predictions)

            # Calculate metrics
            aligned_prices = prices[-len(predictions):]  # Take
