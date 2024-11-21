import streamlit as st
import os
import yfinance as yf
from backtesting import Backtest, Strategy
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt
from bokeh.models import DatetimeTickFormatter
from bokeh.io import output_file, show
from bokeh.plotting import figure
import pandas as pd

# Helper function to fetch stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

# Define strategies
class DDPGStrategy(Strategy):
    def init(self):
        self.agent = st.session_state.agent  # Use the trained DDPG agent

    def next(self):
        state = [
            self.equity,
            self.position.size,
            self.data.Open[-1],
            self.data.High[-1],
            self.data.Low[-1],
            self.data.Close[-1],
        ]
        action = self.agent.get_action(state)
        if action > 0.5 and not self.position:
            self.buy()
        elif action <= 0.5 and self.position:
            self.sell()


class PPOStrategy(Strategy):
    def init(self):
        self.agent = st.session_state.agent  # Use the trained PPO agent

    def next(self):
        state = [
            self.equity,
            self.position.size,
            self.data.Open[-1],
            self.data.High[-1],
            self.data.Low[-1],
            self.data.Close[-1],
        ]
        action, _ = self.agent.get_action(state)  # Get action from the agent
        if action == 0 and not self.position:
            self.buy()
        elif action == 1 and self.position:
            self.sell()



# Function to train DDPG model
def train_ddpg(data, hyperparams, epochs):
    agent = DDPGAgent(input_dim=6, action_dim=1, **hyperparams)
    for epoch in range(epochs):
        # Simulated training loop
        for i in range(len(data) - 1):
            state = [
                data.iloc[i]["Close"],
                0,  # Dummy position size
                data.iloc[i]["Open"],
                data.iloc[i]["High"],
                data.iloc[i]["Low"],
                data.iloc[i]["Close"],
            ]
            next_state = [
                data.iloc[i + 1]["Close"],
                0,  # Dummy position size
                data.iloc[i + 1]["Open"],
                data.iloc[i + 1]["High"],
                data.iloc[i + 1]["Low"],
                data.iloc[i + 1]["Close"],
            ]
            reward = next_state[0] - state[0]  # Reward = next close - current close
            done = i == len(data) - 2
            agent.train_step(state, reward, next_state, done)
    return agent


# Function to train PPO model
def train_ppo(data, hyperparams, epochs):
    agent = PPOAgent(input_dim=6, action_dim=2, **hyperparams)
    for epoch in range(epochs):
        # Simulated training loop
        for i in range(len(data) - 1):
            state = [
                data.iloc[i]["Close"],
                0,  # Dummy position size
                data.iloc[i]["Open"],
                data.iloc[i]["High"],
                data.iloc[i]["Low"],
                data.iloc[i]["Close"],
            ]
            next_state = [
                data.iloc[i + 1]["Close"],
                0,  # Dummy position size
                data.iloc[i + 1]["Open"],
                data.iloc[i + 1]["High"],
                data.iloc[i + 1]["Low"],
                data.iloc[i + 1]["Close"],
            ]
            reward = next_state[0] - state[0]  # Reward = next close - current close
            done = i == len(data) - 2
            agent.train_step(state, reward, next_state, done)
    return agent


# Function to run backtesting
def run_backtesting(strategy, data, cash=10000, commission=0.002):
    bt = Backtest(data, strategy, cash=cash, commission=commission)
    stats = bt.run()
    html_file = f"{strategy.__name__}_backtest_report.html"
    bt.plot(open_browser=False, filename=html_file)  # Save HTML report
    return bt, stats, html_file


# Initialize Streamlit app
st.set_page_config(page_title="Trading Strategy App", layout="wide")
st.title("Stock Trading Strategies with Backtesting")
st.markdown("Train a strategy, backtest it, and visualize results!")

# Session State Initialization
if "data" not in st.session_state:
    st.session_state.data = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "strategy" not in st.session_state:
    st.session_state.strategy = None
if "hyperparams" not in st.session_state:
    st.session_state.hyperparams = {"hidden_dim": 64, "lr": 0.001, "gamma": 0.99}
if "epochs" not in st.session_state:
    st.session_state.epochs = 50

# Sidebar for strategy selection and hyperparameter tuning
with st.sidebar:
    st.header("Strategy Configuration")
    # Strategy selection
    strategy_options = {"DDPG": DDPGStrategy, "PPO": PPOStrategy}
    strategy_choice = st.selectbox("Choose Strategy", options=strategy_options.keys())
    st.session_state.strategy = strategy_options[strategy_choice]

    # Hyperparameter inputs
    st.subheader("Hyperparameters")
    st.session_state.hyperparams["hidden_dim"] = st.number_input(
        "Hidden Dimension", value=64, min_value=16, max_value=512
    )
    st.session_state.hyperparams["lr"] = st.number_input(
        "Learning Rate", value=0.001, format="%.6f", min_value=0.00001, max_value=0.1
    )
    st.session_state.hyperparams["gamma"] = st.number_input(
        "Discount Factor (γ)", value=0.99, format="%.2f", min_value=0.5, max_value=1.0
    )
    st.session_state.epochs = st.number_input(
        "Training Epochs", value=50, min_value=10, max_value=500
    )

# Main content
ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS):", value="GOOG")
period = st.selectbox("Select Data Period", options=["1y", "6mo", "3mo"], index=0)

if st.button("Load Data"):
    try:
        data = get_stock_data(ticker, period=period)
        
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        
        st.session_state.data = data
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if st.session_state.data is not None:
    data = st.session_state.data
    st.write(data.head())
    
    # Ensure the index is in datetime format before plotting
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)  # Convert to datetime if not already
        
    # Show the data graph
    st.write("### Stock Data Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Close'], label="Close Price", color="blue")
    ax.set_title(f"{ticker} Stock Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Train Model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                if strategy_choice == "DDPG":
                    agent = train_ddpg(data, st.session_state.hyperparams, st.session_state.epochs)
                elif strategy_choice == "PPO":
                    agent = train_ppo(data, st.session_state.hyperparams, st.session_state.epochs)
                st.session_state.agent = agent
                st.session_state.trained_model = agent
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error during training: {e}")

    if st.button("Backtest Model"):
        if "trained_model" not in st.session_state:
            st.error("No trained model found. Please train a model first.")
        elif not data.empty:
            try:
                # Define strategy for backtesting.py
                class CustomStrategy(Strategy):
                    def init(self):
                        self.agent = st.session_state.trained_model

                    def next(self):
                        # Prepare state for the agent
                        state = [
                            self.equity,
                            self.position.size,
                            self.data.Open[-1],
                            self.data.High[-1],
                            self.data.Low[-1],
                            self.data.Close[-1],
                        ]
                        
                        # Get action from the trained model
                        action = self.agent.get_action(state)
                        
                        # Map action to buy, sell, or hold
                        if action >= 0.5:  # Threshold to trigger a buy
                            if not self.position:  # Buy if no position exists
                                self.buy()
                        else:  # If action < 0.5, trigger a sell
                            if self.position:  # Sell if position exists
                                self.sell()
                        # print what the model is doing
                        print(f"Action: {action}, Position: {self.position.size}, Equity: {self.equity}")

                # Prepare data for backtesting
                df_bt = data[["Open", "High", "Low", "Close", "Volume"]].copy()

                # Run backtest
                bt = Backtest(df_bt, CustomStrategy, cash=10000, commission=0.002)
                stats = bt.run()

                # Display stats
                st.write("### Backtest Results")
                st.dataframe(stats)

                # Save the backtesting plot
                html_file_path = "backtest_report.html"
                bt.plot(filename=html_file_path)#, open_browser=False)

                # Display a download link for the HTML file
                st.success("Backtest completed!")
                st.markdown(
                    f'<a href="file://D:/C_Drive/Desktop/CS/USAR/MinorProject/trading/{html_file_path}" target="_blank">Open Backtesting Report</a>',
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Error during backtesting: {e}")
        else:
            st.error("No data available for backtesting.")

# Button to clear session state
if st.button("Clear Session State"):
    st.session_state.clear()
    st.success("Session state cleared!")
