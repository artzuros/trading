import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, initial_cash=10000, lookback=5, action_space_type="discrete"):
        super(TradingEnv, self).__init__()

        # Store data and parameters
        self.data = data.values
        self.initial_cash = initial_cash
        self.lookback = lookback
        self.action_space_type = action_space_type

        # Define action space
        if action_space_type == "discrete":
            self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        elif action_space_type == "continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # -1: Max sell, 1: Max buy
        else:
            raise ValueError("Invalid action_space_type. Choose 'discrete' or 'continuous'.")

        # Observation space: OHLC history + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback * 4 + 3,), dtype=np.float32
        )

        # Initialize variables
        self.reset()

    def reset(self):
        """Reset environment to the initial state."""
        self.current_step = self.lookback
        self.cash = self.initial_cash
        self.shares_held = 0
        self.done = False

        # Portfolio value: cash + value of held shares
        self.portfolio_value = self.cash
        self.previous_portfolio_value = self.portfolio_value

        return self._get_observation()

    def _get_observation(self):
        """Construct the observation."""
        ohlc_history = self.data[self.current_step - self.lookback : self.current_step, :].astype(np.float32)
        portfolio_state = np.array([self.cash, self.shares_held, self.portfolio_value], dtype=np.float32)
        return np.concatenate((ohlc_history.flatten(), portfolio_state))

    def step(self, action):
        """Execute one step in the environment."""
        if self.done:
            raise RuntimeError("Step called on a finished environment. Reset the environment first.")

        current_price = self.data[self.current_step, 3]  # Use 'Close' price for trading

        if self.action_space_type == "discrete":
            # 0: Hold, 1: Buy, 2: Sell
            if action == 1:  # Buy all available with cash
                self._buy_shares(current_price, fraction=1.0)
            elif action == 2:  # Sell all shares held
                self._sell_shares(current_price, fraction=1.0)

        elif self.action_space_type == "continuous":
            # Continuous action: -1 (sell all) to +1 (buy all)
            action_value = np.clip(action[0], -1, 1)  # Extract scalar from array
            if action_value > 0:  # Buy proportion
                self._buy_shares(current_price, fraction=action_value)
            elif action_value < 0:  # Sell proportion
                self._sell_shares(current_price, fraction=-action_value)

        # Update portfolio value
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.shares_held * current_price

        # Calculate reward
        reward = self.portfolio_value - self.previous_portfolio_value

        # Move to the next step
        self.current_step += 1

        # Check if we're done
        if self.current_step >= len(self.data) - 1 or self.portfolio_value <= 0:
            self.done = True

        # Create the next observation
        observation = self._get_observation()

        # Debug prints
        print(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, Portfolio Value: {self.portfolio_value}")

        return observation, reward, self.done, {}

    def _buy_shares(self, price, fraction):
        """Buy a fraction of shares with available cash."""
        if price <= 0 or fraction <= 0:
            return
        amount_to_invest = self.cash * fraction
        shares_to_buy = amount_to_invest // price
        self.cash -= shares_to_buy * price
        self.shares_held += shares_to_buy

    def _sell_shares(self, price, fraction):
        """Sell a fraction of shares held."""
        if price <= 0 or fraction <= 0:
            return
        shares_to_sell = int(self.shares_held * fraction)
        self.cash += shares_to_sell * price
        self.shares_held -= shares_to_sell

    def render(self):
        """Render the current state."""
        print(f"Step: {self.current_step}")
        print(f"Price: {self.data[self.current_step, 3]} (Close)")
        print(f"Cash: {self.cash}")
        print(f"Shares Held: {self.shares_held}")
        print(f"Portfolio Value: {self.portfolio_value}")
