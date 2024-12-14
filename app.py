import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_and_backtest import fetch_ohlc_data, main as train_and_backtest_main
from env import TradingEnv
from argparse import Namespace

# Configure Streamlit app
st.title("RL-based Stock Trading Simulator")

# User inputs
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
model = st.selectbox("Select Model", ["dqn", "ddpg", "a2c"])
episodes = st.slider("Training Episodes", min_value=10, max_value=1000, value=500)
epsilon = st.slider("Epsilon (DQN)", min_value=0.01, max_value=1.0, value=0.1) if model == "dqn" else None
seed = st.number_input("Random Seed", value=42)

discrete_action_space = True if model == "dqn" else False

# Trigger training and backtesting
if st.button("Run Simulation"):
    st.write("Fetching data and training the model. Please wait...")
    ohlc_data = fetch_ohlc_data(ticker, str(start_date), str(end_date))
    
    # Save inputs for backend script
    args = Namespace(
        ticker=ticker,
        start_date=str(start_date),
        end_date=str(end_date),
        model=model,
        episodes=episodes,
        epsilon=epsilon,
        discrete=discrete_action_space,
        seed=seed
    )

    train_and_backtest_main(args)
    st.write("Simulation complete! Plotting results...")

    # Fetch saved results
    train_rewards = np.load(f"{model}_train_rewards.npy")
    actions = np.load(f"{model}_backtest_actions.npy")
    ohlc = ohlc_data.reset_index()

    # Plot training rewards
    st.subheader("Training Rewards Over Episodes")
    st.line_chart(train_rewards)

    # Backtest results visualization
    st.subheader("Backtest Results")
    buy_sell_points = [(i, action) for i, action in enumerate(actions) if action != 0]
    profits = []
    for i, action in buy_sell_points:
        price = ohlc.loc[i, "Close"]
        action_type = "Buy" if action == 1 else "Sell"
        profits.append(price if action == 1 else -price)
        # st.write(f"Step {i}: {action_type} at ${price:.2f}")

    # Debug prints
    st.write("Buy/Sell Points:", buy_sell_points)

    # Plot prices with buy/sell points
    plt.figure(figsize=(10, 5))
    plt.plot(ohlc["Close"], label="Close Price")

    # Plot buy points
    buy_points = [(i, ohlc.iloc[i]["Close"]) for i, action in buy_sell_points if action == 1]
    if buy_points:
        # st.write("Buy Points:", buy_points)
        buy_indices, buy_prices = zip(*buy_points)
        plt.scatter(buy_indices, buy_prices, color="green", marker="^", label="Buy")

    # Plot sell points
    sell_points = [(i, ohlc.iloc[i]["Close"]) for i, action in buy_sell_points if action == 2]
    if sell_points:
        # st.write("Sell Points:", sell_points)
        sell_indices, sell_prices = zip(*sell_points)
        plt.scatter(sell_indices, sell_prices, color="red", marker="v", label="Sell")

    plt.title("Backtest: Prices with Buy/Sell Points")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    # st.write(f"Total Profit: ${sum(profits):.2f}")