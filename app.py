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
    # Set the title and introduction for the Streamlit app
    st.title("📈 Stock Analysis & Prediction App")
    st.markdown("""
    This app provides comprehensive stock analysis, price predictions, and risk metrics.
    Enter a stock ticker symbol to get started!
    """)

    # Create input columns for stock ticker and years of historical data
    col1, col2 = st.columns([2, 1])
    with col1:
        # Input for stock ticker, converting to uppercase and removing whitespace
        ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "").strip().upper()
    with col2:
        # Number input for years of historical data with validation
        years = st.number_input("Years of Historical Data", min_value=1, max_value=20, value=10)

    # Only proceed if a valid ticker is entered
    if ticker:
        try:
            # Initialize the StockPredictor with the chosen ticker and years
            predictor = StockPredictor(ticker, years)

            # Show a loading spinner while fetching data
            with st.spinner(f'Fetching data for {ticker}...'):
                # Fetch historical stock data
                predictor.fetch_data()

            # Get stock metrics from Yahoo Finance
            stock_metrics = get_stock_metrics(ticker)

            # Calculate current price and daily change
            current_price = predictor.data['Close'].iloc[-1]
            daily_change = (predictor.data['Close'].iloc[-1] - predictor.data['Close'].iloc[-2]) / \
                           predictor.data['Close'].iloc[-2] * 100

            # Display current stock information in metrics
            st.markdown("### Current Stock Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Daily Change", f"{daily_change:.2f}%", f"{daily_change:.2f}%")
            col3.metric("Trading Volume", f"{predictor.data['Volume'].iloc[-1]:,.0f}")

            # Display company metrics in styled cards
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

            # Generate and display technical analysis charts
            st.markdown("### Technical Analysis Charts")
            figs = predictor.create_plots()
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)

            # Prediction section
            st.markdown("### Price Predictions")
            
            prediction_models = ['Term Adjusted Exponential Smoothening', 'Long Short-Term Memory', 'Recurrent Neural Networks', 'Auto-Regressive Integrated Moving Averages']
            selected_model = st.selectbox("Select Prediction Model", prediction_models)
            days = st.number_input("Number of days", min_value=1, value=5, max_value=10)
            
            if st.button("Predict Stock Prices"):
                with st.spinner(f"Training {selected_model} model and generating predictions for next {days} business days..."):
                    # Initialize predictions DataFrame
                    all_predictions = pd.DataFrame(columns=['Date', 'Predicted Price'])
                    
                    try:
                        if selected_model == 'Term Adjusted Exponential Smoothening':
                            predictor.train_taes_model()
                            predictions = predictor.predict_future(days=days, model='TAES')
                            model_pred_df = pd.DataFrame({
                                'Date': predictions.index.strftime('%Y-%m-%d'),
                                'Predicted Price': predictions.values
                                })
                            #model_pred_df['Model'] = selected_model
                            all_predictions = pd.concat([all_predictions, model_pred_df], ignore_index=True)
                                
                        elif selected_model == 'Long Short-Term Memory':
                            predictor.train_lstm_model()
                            predictions = predictor.predict_future(days=days, model=selected_model)
                            model_pred_df = pd.DataFrame({
                                'Date': predictions.index.strftime('%Y-%m-%d'),
                                'Predicted Price': predictions.values
                                })
                            all_predictions = pd.concat([all_predictions, model_pred_df], ignore_index=True)
                                
                        elif selected_model == 'Recurrent Neural Networks':
                            predictor.train_rnn_model()
                            predictions = predictor.predict_future(days=days, model=selected_model)
                            model_pred_df = pd.DataFrame({
                                'Date': predictions.index.strftime('%Y-%m-%d'),
                                'Predicted Price': predictions.values
                            })
                            #model_pred_df['Model'] = selected_model
                            all_predictions = pd.concat([all_predictions, model_pred_df], ignore_index=True)
                            
                        elif selected_model == 'Auto-Regressive Integrated Moving Averages':  
                            predictor.train_arima_model()
                            
                            predictions = predictor.predict_future(days=days, model='ARIMA')
                            
                            model_pred_df = pd.DataFrame({
                                'Date': predictions.index.strftime('%Y-%m-%d'),
                                'Predicted Price': predictions.values
                            })
                            #model_pred_df['Model'] = selected_model
                            
                            all_predictions = pd.concat([all_predictions, model_pred_df], ignore_index=True)
                    
                    except Exception as e:
                        st.error(f"Error predicting with {selected_model} model: {str(e)}")
                        
                        
                st.markdown("#### Predicted Prices for Next {} Business Days".format(days))
                st.markdown("""
                <div class="prediction-table">
                """, unsafe_allow_html=True)
                st.dataframe(
                    all_predictions.style.format({
                        'Date': lambda x: x,
                        'Predicted Price': '${:.2f}'
                    }).set_properties(**{
                        #'background-color': 'lightskyblue',
                        #'color': 'black'
                    }).highlight_max(
                        subset=['Predicted Price'], color='#2b6929'
                    ),
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Risk Analysis section
            st.markdown("### Risk Analysis")
            # Input number of shares for risk calculation
            freq_dict = ['Years', 'Months', 'Weeks', 'Days']
            n_shares = st.number_input("Number of Shares", min_value=1, value=100, max_value=5000)
            col1, col2 = st.columns([2, 1])
            with col1:
                freq = st.number_input("Enter the holding period", min_value=1.0, value=1.0, max_value=500.0, step=1.0)

            with col2:
                freq_mode = st.selectbox("Select frequency mode", freq_dict)

            if freq_mode == 'Years':
                holding_period = freq * 365

            elif freq_mode == 'Months':
                holding_period = freq * 30

            elif freq_mode == 'Weeks':
                holding_period = freq * 7

            else:
                holding_period = freq

            
            if st.button("Calculate Risk Metrics"):


                with st.spinner("Calculating Value at Risk..."):
                    # Calculate VaR metrics
                    var_metrics = predictor.calculate_var(n_shares=n_shares, holding_period=holding_period)

                    # Prepare VaR data for display
                    var_data = []
                    methods = ['Parametric', 'Historical', 'Monte Carlo', 'Benchmark']

                    for method in methods:
                        # Safely handle VaR value retrieval
                        var_value = abs(var_metrics.get(f'{method.replace(" ", "_")}_VaR', 0))

                        # Safely handle required capital
                        if method != 'Benchmark':
                            required_capital = var_metrics['Required_Capital'].get(method, 0)
                        else:
                            required_capital = 0

                        var_data.append({
                            'Method': method,
                            'VaR': var_value,
                            'Required Capital': required_capital
                        })

                    var_df = pd.DataFrame(var_data)

                    # Display VaR metrics in a styled table
                    st.markdown("#### Value at Risk (VaR) Analysis")
                    st.markdown("""
                            <div class="prediction-table">
                            """, unsafe_allow_html=True)
                    st.dataframe(
                        var_df.style.format({
                            'VaR': '${:,.2f}',
                            'Required Capital': '${:,.2f}'
                        }).set_properties(**{
                            #'background-color': 'lightyellow',
                            #'color': 'black'
                        }).highlight_min(
                            subset=['VaR'], color='#2b6929'
                        ),
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            # Handle and display any errors that occur during processing
            st.error(f"Error: {str(e)}")

def display_portfolio_dashboard():
    st.title("Portfolio Analysis")
    
    # Initialize portfolio data in session state
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = []
    
    # Form for adding stocks
    with st.form(key="portfolio_form"):
        symbol = st.text_input(f"Stock Symbol {len(st.session_state.portfolio_data) + 1}")
        shares = st.number_input(f"Number of Shares {len(st.session_state.portfolio_data) + 1}", min_value=1, step=1)
        allocation = st.number_input(f"Allocation (%) {len(st.session_state.portfolio_data) + 1}", min_value=0.0, max_value=100.0, step=0.1)
        submit_button = st.form_submit_button("Add to Portfolio")
        
        if submit_button and symbol:
            st.session_state.portfolio_data.append({"symbol": symbol, "shares": shares, "allocation": allocation})
    
    # Display current portfolio
    if st.session_state.portfolio_data:
        portfolio_df = pd.DataFrame(st.session_state.portfolio_data)
        st.dataframe(portfolio_df)
    
    # Analyze button
    analyze_button = st.button("Analyze")
    if analyze_button and st.session_state.portfolio_data:
        # Calculate portfolio performance metrics
        portfolio_return = calculate_portfolio_return(st.session_state.portfolio_data)
        portfolio_std_dev = calculate_portfolio_std_dev(st.session_state.portfolio_data)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_std_dev)
        treynor_ratio = calculate_treynor_ratio(portfolio_return, st.session_state.portfolio_data)
        sortino_ratio = calculate_sortino_ratio(portfolio_return, st.session_state.portfolio_data)
        information_ratio = calculate_information_ratio(portfolio_return, st.session_state.portfolio_data)
        jensen_alpha = calculate_jensen_alpha(portfolio_return, st.session_state.portfolio_data)
        
        # Display performance metrics
        st.markdown("### Portfolio Performance Metrics")
        metrics_data = {
            "Metric": ["Portfolio Return", "Portfolio Standard Deviation", "Sharpe's Ratio", "Treynor's Ratio", "Sortino's Ratio", "Information Ratio", "Jensen's Alpha"],
            "Value": [portfolio_return, portfolio_std_dev, sharpe_ratio, treynor_ratio, sortino_ratio, information_ratio, jensen_alpha]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({
            "Value": "{:.2f}"
        }))

# Main execution block
if __name__ == "__main__":
    app_mode = st.selectbox("Select Mode", ["Single Stock", "Portfolio"])

if app_mode == "Single Stock":
    main()
elif app_mode == "Portfolio":
    display_portfolio_dashboard()

