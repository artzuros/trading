import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
from gymnasium import spaces, Env
from stable_baselines3 import PPO, DQN

# TODO DDPG

# Define the custom strategy
class periodicStrategy(Strategy):
    def init(self):
        print(f"Start with equity={self.equity:.2f}")
        
    def next(self, action:int|None=None):
        print(f"Action={action} Equity={self.equity:.2f} Date={self.data.index[-1]}")
        if action:
            if action == 1:
                self.buy()
            elif action == 2:
                self.position.close()
            
    def observation(self):
        closes = self.data.Close[-20:]
        closes = (closes - closes.min()) / (closes.max() - closes.min())
        return [closes]

# Define the custom environment
class CustomEnv(Env):
    """Custom Environment that follows gym interface."""    
    def __init__(self, bt: Backtest):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 20), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Actions: 0 = do nothing, 1 = buy, 2 = sell
        self.bt = bt
        
    def reward_calculation(self):
        if self.previous_equity < self.bt._step_strategy.equity:
            return +1
        return -1
        
    def check_done(self):
        if self.bt._step_time + 2 > len(self.bt._data):
            self.render()
            return True
        return False
        
    def step(self, action):
        obs = self.bt._step_strategy.observation()
        reward = self.reward_calculation()
        done = self.check_done()
        info = {}
        self.bt.next(action=action)
        return obs, reward, False, done, info

    def reset_backtesting(self):
        self.bt.initialize()
        self.bt.next()
        while True:
            obs = self.bt._step_strategy.observation()
            if np.shape(obs) == (1, 20):  # Wait for the first valid observation
                break
            self.bt.next()
                
    def reset(self, seed=None):
        self.previous_equity = 10
        self.reset_backtesting()
        return self.bt._step_strategy.observation(), {}

    def render(self, mode='human'):
        result = self.bt.next(done=True)
        self.bt.plot(results=result, open_browser=False)
        
    def close(self):
        pass

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def generate_sin_wave(periods: int = 1000, amplitude: float = 1.0) -> pd.DataFrame:
    x = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    y = amplitude * pd.Series(data=np.sin(np.linspace(0, 10 * np.pi, periods)), index=x) + 2
    # Create a DataFrame with the required columns
    data = pd.DataFrame({'Open': y, 'High': y, 'Low': y, 'Close': y, 'Volume': y})
    return data

# Fetch real data for Apple (AAPL) from Yahoo Finance
stock_data = get_stock_data("AAPL")
print(stock_data.head())  # Check the first few rows

data = generate_sin_wave()
print(data)

# Instantiate the backtest with the fetched real data
bt = Backtest(stock_data, periodicStrategy, cash=10000)

# Instantiate the custom environment with the backtest
env = CustomEnv(bt)

# Train PPO agent (you can replace PPO with DDPG or DQN as shown earlier)
# ppo_model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./logs_ppo/")
# ppo_model.learn(total_timesteps=1000, log_interval=1)

# # Alternatively, use DQN
dqn_model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./logs_dqn/")
dqn_model.learn(total_timesteps=10000, log_interval=1)

# Optionally save any of the models
# ppo_model.save("ppo_stock_trader")
# ddpg_model.save("ddpg_stock_trader")
# dqn_model.save("dqn_stock_trader")
