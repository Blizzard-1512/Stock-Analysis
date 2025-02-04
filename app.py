import self
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

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

class PortfolioManager:
    def __init__(self):
        """Initialize portfolio in session state if not exists"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
            
    def add_stock(self, symbol: str, units: int) -> None:
        """
        Add a stock to the portfolio
        Args:
            symbol (str): Stock ticker symbol
            units (int): Number of shares/units
        """
        symbol = symbol.upper().strip()
        if symbol and not any(s['symbol'] == symbol for s in st.session_state.portfolio):
            st.session_state.portfolio.append({
                'symbol': symbol,
                'units': int(units)
            })
            
    def remove_stock(self, symbol: str) -> None:
        """Remove a stock from the portfolio"""
        st.session_state.portfolio = [
            s for s in st.session_state.portfolio 
            if s['symbol'] != symbol.upper()
        ]
        
    def get_portfolio_data(self) -> pd.DataFrame:
        """
        Retrieve current portfolio data with pricing information
        Returns:
            pd.DataFrame: Portfolio data with current values
        """
        portfolio_data = []
        
        for item in st.session_state.portfolio:
            try:
                # Get current price from Yahoo Finance
                stock = yf.Ticker(item['symbol'])
                hist = stock.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_value = current_price * item['units']
                    
                    portfolio_data.append({
                        'Symbol': item['symbol'],
                        'Units': item['units'],
                        'Price': current_price,
                        'Value': current_value,
                        'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    st.error(f"No data available for {item['symbol']}")
                    
            except Exception as e:
                st.error(f"Error fetching data for {item['symbol']}: {str(e)}")
        
        return pd.DataFrame(portfolio_data)
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        df = self.get_portfolio_data()
        return df['Value'].sum() if not df.empty else 0.0
    
    def clear_portfolio(self) -> None:
        """Reset portfolio to empty state"""
        st.session_state.portfolio = []
    
    def portfolio_exists(self) -> bool:
        """Check if portfolio contains any holdings"""
        return len(st.session_state.portfolio) > 0
    
    def get_stock_symbols(self) -> list:
        """Get list of symbols in portfolio"""
        return [item['symbol'] for item in st.session_state.portfolio]

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
            from statsmodels.tsa.arima.model import ARIMA
            
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
            aligned_prices = prices[-len(predictions):]  # Take only the last 'len(predictions)' price 
            if len(aligned_prices) != len(predictions):
                raise ValueError("Inconsistent lengths for metrics calculation.")
            mape = mean_absolute_percentage_error(aligned_prices, predictions)
            rmse = np.sqrt(mean_squared_error(aligned_prices, predictions))

            # Store model and metrics
            self.models['ARIMA'] = {
                'model': model,
                'metrics': {
                    'MAPE': mape,
                    'RMSE': rmse,
                    'Method': 'ARIMA'
                }
            }

            return self.models['ARIMA']['metrics']

        except ImportError:
            st.warning("Statsmodels not available. Skipping ARIMA model.")
            return None


    def calculate_var(self, confidence_level: float = 0.99, holding_period: int = 1, n_shares: int = 100) -> dict:
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        returns = self.data['Returns'].dropna()
        current_price = self.data['Close'].iloc[-1]
        position_value = current_price * n_shares

        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)

        # Calculate VaR values
        parametric_var = position_value * (
                z_score * std_return * np.sqrt(holding_period) - mean_return * holding_period)
        historical_var = position_value * returns.quantile(1 - confidence_level) * np.sqrt(holding_period)

        # Monte Carlo VaR
        n_simulations = 10000
        np.random.seed(42)
        mc_returns = np.random.normal(mean_return, std_return, n_simulations)
        mc_var = position_value * np.percentile(mc_returns, (1 - confidence_level) * 100) * np.sqrt(holding_period)

        # Calculate required capital (3x VaR)
        capital_multiplier = 3
        required_capital = {
            'Parametric': abs(float(parametric_var)) * capital_multiplier,
            'Historical': abs(float(historical_var)) * capital_multiplier,
            'Monte Carlo': abs(float(mc_var)) * capital_multiplier
        }

        benchmark_var = position_value * 0.015  # 1.5% of position value

        self.var_metrics = {
            'Parametric_VaR': float(parametric_var),
            'Historical_VaR': float(historical_var),
            'Monte_Carlo_VaR': float(mc_var),
            'Benchmark_VaR': float(benchmark_var),
            'Required_Capital': required_capital
        }

        return self.var_metrics

    def predict_future(self, days: int = 5, model: str = 'TAES'):
        """
        Predict future stock prices using specified model
        """
        if model not in self.models or self.models[model] is None:
            raise ValueError(f"Model {model} not trained. Train the model first.")

        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=days,
                                     freq='B')

        # Prediction logic for different models
        if model == 'TAES':
            # Existing trend-adjusted method
            taes_model = self.models['TAES']
            predictions = np.array([taes_model['last_train_price'] + (i + 1) * 
                                    (taes_model['avg_daily_change'] + taes_model['trend']) 
                                    for i in range(days)])
        
        elif model in ['LSTM', 'RNN']:
            # For neural network models
            ml_model = self.models[model]['model']
            scaler = self.models[model]['scaler']
            
            # Get the last sequence of prices
            last_sequence = self.data['Close'].tail(self.prepare_ml_data()['look_back'])
            last_sequence_scaled = scaler.transform(last_sequence.values.reshape(-1, 1))
            
            # Predict future prices
            predictions = []
            current_sequence = last_sequence_scaled.reshape(1, -1, 1)
            
            for _ in range(days):
                # Predict next price
                next_pred_scaled = ml_model.predict(current_sequence)
                next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
                
                predictions.append(next_pred)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = next_pred_scaled
        
        elif model == 'ARIMA':
            # ARIMA prediction
            arima_model = self.models['ARIMA']['model']
            if days > len(self.data):
                raise ValueError("Prediction period exceeds available data length.")
            predictions = arima_model.forecast(steps=days)

        # Convert to Series
        self.model_predictions[model] = pd.Series(predictions, index=future_dates)
        return self.model_predictions[model]

    def create_plots(self):
        if self.data is None:
            raise ValueError("No data available.")

        # Create figures list
        figs = []

        # OHLC plot
        fig_ohlc = go.Figure()
        fig_ohlc.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='OHLC'
            )
        )
        fig_ohlc.update_layout(
            title=f"{self.ticker} OHLC Chart",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        figs.append(fig_ohlc)

        # Moving Averages plot
        fig_ma = go.Figure()
        for ma, color in zip(['MA5', 'MA20', 'MA50'], ['gold', 'orange', 'red']):
            fig_ma.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data[ma],
                    name=f'{ma}',
                    line=dict(color=color, width=1)
                )
            )
        fig_ma.update_layout(
            title=f"{self.ticker} Moving Averages",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        figs.append(fig_ma)

        # Bollinger Bands plot
        fig_bb = go.Figure()
        fig_bb.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Upper_Band'],
                name='Upper Band',
                line=dict(color='gray', width=1)
            )
        )
        #fig_bb.add_trace(
            #go.Scatter(
                #x=self.data.index,
                #y=self.data['MA20'],
                #name='MA20',
                #line=dict(color='blue', width=1)
            #)
        #)
        fig_bb.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Lower_Band'],
                name='Lower Band',
                line=dict(color='gray', width=1),
                fill='tonexty'
            )
        )
        fig_bb.update_layout(
            title=f"{self.ticker} Bollinger Bands",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        figs.append(fig_bb)

        # Volume plot
        colors = ['red' if row['Open'] - row['Close'] > 0
                  else 'green' for index, row in self.data.iterrows()]

        fig_volume = go.Figure()
        fig_volume.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            )
        )
        fig_volume.update_layout(
            title=f"{self.ticker} Trading Volume",
            yaxis_title="Volume",
            template="plotly_white",
            height=500
        )
        figs.append(fig_volume)

        # Volatility plot
        fig_volatility = go.Figure()
        fig_volatility.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Volatility'],
                name='Volatility',
                line=dict(color='purple', width=1),
                fill='tozeroy'
            )
        )
        fig_volatility.update_layout(
            title=f"{self.ticker} Price Volatility",
            yaxis_title="Volatility",
            template="plotly_white",
            height=500
        )
        figs.append(fig_volatility)

        return figs

def get_stock_metrics(ticker):
    """
    Fetch key stock metrics from Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics with fallback values
        metrics = {
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
            'EPS': info.get('trailingEps', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Sector': info.get('sector', 'N/A')
        }

        # Format market cap
        if isinstance(metrics['Market Cap'], (int, float)):
            if metrics['Market Cap'] >= 1e12:
                metrics['Market Cap'] = f'${metrics["Market Cap"]/1e12:.2f}T'
            elif metrics['Market Cap'] >= 1e9:
                metrics['Market Cap'] = f'${metrics["Market Cap"]/1e9:.2f}B'
            else:
                metrics['Market Cap'] = f'${metrics["Market Cap"]/1e6:.2f}M'

        return metrics
    except Exception as e:
        st.error(f"Could not fetch stock metrics: {e}")
        return {}


def main():
    # Page configuration
    st.title("📈 Smart Stock Analyzer Pro")
    
    # Mode selection
    analysis_mode = st.selectbox("Analysis Mode", 
                               ["Single Stock", "Portfolio Analysis"], 
                               index=0)

    if analysis_mode == "Single Stock":
        # Original single stock interface
        st.markdown("""
        This app provides comprehensive stock analysis, price predictions, and risk metrics.
        Enter a stock ticker symbol to get started!
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "").strip().upper()
        with col2:
            years = st.number_input("Years of Historical Data", min_value=1, max_value=20, value=10)

        if ticker:
            try:
                predictor = StockPredictor(ticker, years)
                with st.spinner(f'Fetching data for {ticker}...'):
                    predictor.fetch_data()

                # Display current price information
                current_price = predictor.data['Close'].iloc[-1]
                daily_change = (predictor.data['Close'].iloc[-1] - predictor.data['Close'].iloc[-2]) / \
                             predictor.data['Close'].iloc[-2] * 100
                
                st.markdown("### Current Stock Information")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Daily Change", f"{daily_change:.2f}%")
                col3.metric("Trading Volume", f"{predictor.data['Volume'].iloc[-1]:,.0f}")

                # Display company metrics
                stock_metrics = get_stock_metrics(ticker)
                st.markdown("### Company Metrics")
                metrics_cols = st.columns(4)
                metric_keys = list(stock_metrics.keys())
                for i, key in enumerate(metric_keys):
                    with metrics_cols[i % 4]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{key}</h3>
                            <div class="value">{stock_metrics[key]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Display technical charts
                st.markdown("### Technical Analysis Charts")
                figs = predictor.create_plots()
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)

                # Prediction section
                st.markdown("### Price Predictions")
                prediction_models = ['Term Adjusted Exponential Smoothening', 
                                   'Long Short-Term Memory', 
                                   'Recurrent Neural Networks', 
                                   'Auto-Regressive Integrated Moving Averages']
                selected_model = st.selectbox("Select Prediction Model", prediction_models)
                days = st.number_input("Number of days", min_value=1, value=5, max_value=10)
                
                if st.button("Predict Stock Prices"):
                    with st.spinner(f"Training {selected_model} model..."):
                        all_predictions = pd.DataFrame(columns=['Date', 'Predicted Price'])
                        try:
                            model_map = {
                                'Term Adjusted Exponential Smoothening': 'TAES',
                                'Long Short-Term Memory': 'LSTM',
                                'Recurrent Neural Networks': 'RNN',
                                'Auto-Regressive Integrated Moving Averages': 'ARIMA'
                            }
                            model_key = model_map[selected_model]
                            
                            if model_key == 'TAES':
                                predictor.train_taes_model()
                            elif model_key == 'LSTM':
                                predictor.train_lstm_model()
                            elif model_key == 'RNN':
                                predictor.train_rnn_model()
                            elif model_key == 'ARIMA':
                                predictor.train_arima_model()
                                
                            predictions = predictor.predict_future(days=days, model=model_key)
                            model_pred_df = pd.DataFrame({
                                'Date': predictions.index.strftime('%Y-%m-%d'),
                                'Predicted Price': predictions.values
                            })
                            st.markdown("#### Predicted Prices")
                            st.dataframe(
                                model_pred_df.style.format({
                                    'Predicted Price': '${:.2f}'
                                }).highlight_max(color='#2b6929'),
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")

                # Risk Analysis
                st.markdown("### Risk Analysis")
                n_shares = st.number_input("Number of Shares", min_value=1, value=100, max_value=5000)
                if st.button("Calculate Risk Metrics"):
                    with st.spinner("Calculating Value at Risk..."):
                        try:
                            var_metrics = predictor.calculate_var(n_shares=n_shares)
                            var_data = {
                                'Method': ['Parametric', 'Historical', 'Monte Carlo', 'Benchmark'],
                                'VaR (95%)': [
                                    var_metrics['Parametric_VaR'],
                                    var_metrics['Historical_VaR'],
                                    var_metrics['Monte_Carlo_VaR'],
                                    var_metrics['Benchmark_VaR']
                                ]
                            }
                            st.dataframe(
                                pd.DataFrame(var_data).style.format({'VaR (95%)': '${:,.2f}'}),
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Risk calculation failed: {str(e)}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    else:  
        portfolio_analysis()
        
st.sidebar.markdown("## AI-Powered Tools")
if st.sidebar.button("📰 Market Sentiment Analysis"):
    with st.spinner("Analyzing market sentiment..."):
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
            sample_news = [
                "Tech sector shows strong growth potential",
                "Federal Reserve signals possible rate hikes",
                "Global markets face uncertainty due to geopolitical tensions"
            ]
            results = [{"News": text, 
                        "Sentiment": sentiment_pipeline(text)[0]['label'], 
                        "Confidence": f"{sentiment_pipeline(text)[0]['score']:.2%}"} 
                       for text in sample_news]
            st.sidebar.dataframe(pd.DataFrame(results), use_container_width=True)
        except:
                st.sidebar.error("Sentiment analysis unavailable")

if __name__ == "__main__":
    main()
