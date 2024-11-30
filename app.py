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
        self.model = None
        self.model_fit = None
        self.predictions = None
        self.metrics = {}
        self.var_metrics = {}

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

    def train_model(self, validation_size: int = 30) -> dict:
        try:
            if self.data is None:
                raise ValueError("No data available. Call fetch_data() first.")

            train_data = self.data[:-validation_size]
            validation_data = self.data[-validation_size:]

            last_prices = train_data['Close'].tail(20)
            avg_daily_change = last_prices.diff().mean()
            trend = (last_prices.iloc[-1] - last_prices.iloc[0]) / len(last_prices)
            last_price = last_prices.iloc[-1]
            forecast = np.array([last_price + (i + 1) * (avg_daily_change + trend) for i in range(validation_size)])

            self.last_train_price = train_data['Close'].iloc[-1]
            self.avg_daily_change = train_data['Close'].diff().mean()
            self.trend = (train_data['Close'].iloc[-1] - train_data['Close'].iloc[-20]) / 20

            self.metrics = {
                'MAPE': mean_absolute_percentage_error(validation_data['Close'], forecast),
                'RMSE': np.sqrt(mean_squared_error(validation_data['Close'], forecast)),
                'Method': 'Trend-adjusted exponential smoothing'
            }

            return self.metrics

        except Exception as e:
            st.error(f"Error in training: {str(e)}")
            return None

    def predict_future(self, days: int = 5) -> pd.Series:
        try:
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                         periods=days,
                                         freq='B')

            predictions = np.array([self.last_train_price + (i + 1) * (self.avg_daily_change + self.trend)
                                    for i in range(days)])

            self.predictions = pd.Series(predictions, index=future_dates)
            return self.predictions

        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")

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
        fig_bb.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MA20'],
                name='MA20',
                line=dict(color='blue', width=1)
            )
        )
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
            days = st.number_input("Number of days", min_value=1, value=5, max_value=10)
            if st.button("Predict Stock Prices"):
                with st.spinner("Training model and generating predictions..."):
                    # Train the prediction model
                    predictor.train_model()
                    # Generate future price predictions
                    predictions = predictor.predict_future(days=days)

                    st.markdown("#### Predicted Prices for Next 5 Business Days")
                    pred_df = pd.DataFrame({
                        'Date': predictions.index.strftime('%Y-%m-%d'),
                        'Predicted Price': predictions.values
                    })

                    # Display predictions in a styled table
                    st.markdown("""
                        <div class="prediction-table">
                        """, unsafe_allow_html=True)
                    st.dataframe(
                        pred_df.style.format({
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
            n_shares = st.number_input("Number of Shares", min_value=1, value=100, max_value=5000)
            if st.button("Calculate Risk Metrics"):


                with st.spinner("Calculating Value at Risk..."):
                    # Calculate VaR metrics
                    var_metrics = predictor.calculate_var(n_shares=n_shares)

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
                            subset=['VaR'], color= '#2b6929'
                        ),
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            # Handle and display any errors that occur during processing
            st.error(f"Error: {str(e)}")


# Main execution block
if __name__ == "__main__":
    main()
