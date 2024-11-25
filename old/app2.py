import streamlit as st
import os
import yfinance as yf
from backtesting import Backtest, Strategy
import stable_baselines3 as sb3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from gymnasium import Env


# Create the custom trading environment
class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000  # Starting balance in USD
        self.position = 0  # No position at the start
        self.max_steps = len(data) - 1  # Limit of steps is data length
        self.action_space = spaces.Discrete(3)  # 0 = sell, 1 = hold, 2 = buy
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.balance = 10000
        self.position = 0
        self.current_step = 0
        state = self.get_state()
        return state

    def get_state(self):
        # Return the state with 5 features: [balance, position, current price, high price, low price]
        state = [
            self.balance,
            self.position,
            self.data['Close'].iloc[self.current_step],
            self.data['High'].iloc[self.current_step],
            self.data['Low'].iloc[self.current_step]
        ]
        return np.array(state)

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]

        if action == 0:  # sell
            if self.position > 0:
                self.balance += self.position * current_price  # Close position and sell
                self.position = 0
        elif action == 2:  # buy
            if self.position == 0:
                self.position = self.balance / current_price  # Open position and buy
                self.balance = 0

        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        # Get the next state
        next_state = self.get_state()

        # Simple reward calculation: profit/loss
        reward = self.balance + (self.position * current_price) - 10000  # Profit relative to initial balance

        return next_state, reward, done, {}

    def render(self):
        # You can define this method to display any extra information, but it’s not necessary for backtesting.
        pass


# Helper function to fetch stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


# Define stable-baselines3 strategies
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
        action, _states = self.agent.predict(state, deterministic=True)
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
        action, _states = self.agent.predict(state, deterministic=True)  # Get action from the agent
        if action == 0 and not self.position:
            self.buy()
        elif action == 1 and self.position:
            self.sell()


# Function to train DDPG model using stable-baselines3
def train_ddpg(data, hyperparams, epochs):
    env = DummyVecEnv([lambda: TradingEnv(data)])  # Wrap environment in DummyVecEnv
    agent = DDPG("MlpPolicy", env, **hyperparams)
    agent.learn(total_timesteps=epochs * len(data))
    return agent


# Function to train PPO model using stable-baselines3
def train_ppo(data, hyperparams, epochs):
    env = DummyVecEnv([lambda: TradingEnv(data)])  # Wrap environment in DummyVecEnv
    agent = PPO("MlpPolicy", env, **hyperparams)
    agent.learn(total_timesteps=epochs * len(data))
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
    st.session_state.hyperparams = {"learning_rate": 0.001, "gamma": 0.99}
if "epochs" not in st.session_state:
    st.session_state.epochs = 50

# Sidebar for strategy selection and hyperparameter tuning
with st.sidebar:
    st.header("Strategy Configuration")
    strategy_options = {"DDPG": DDPGStrategy, "PPO": PPOStrategy}
    strategy_choice = st.selectbox("Choose Strategy", options=strategy_options.keys())
    st.session_state.strategy = strategy_options[strategy_choice]

    # Hyperparameter inputs
    st.subheader("Hyperparameters")
    st.session_state.hyperparams["learning_rate"] = st.number_input(
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
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        
        st.session_state.data = data
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if st.session_state.data is not None:
    data = st.session_state.data
    st.write(data.head())
    
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)  # Convert to datetime if not already

    st.write("### Stock Data Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Close'], label="Close Price", color="blue")
    ax.set_title(f"{ticker} Stock Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

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
                class CustomStrategy(Strategy):
                    def init(self):
                        self.agent = st.session_state.agent

                    def next(self):
                        state = [
                            self.equity,
                            self.position.size,
                            self.data.Open[-1],
                            self.data.High[-1],
                            self.data.Low[-1],
                            self.data.Close[-1],
                        ]
                        action, _states = self.agent.predict(state, deterministic=True)
                        if action == 0 and not self.position:
                            self.buy()
                        elif action == 1 and self.position:
                            self.sell()
                
                bt, stats, report = run_backtesting(CustomStrategy, data)
                st.write(f"Backtest Results: {stats}")
                st.write(f"Backtest report saved to: {report}")
            except Exception as e:
                st.error(f"Error during backtesting: {e}")
