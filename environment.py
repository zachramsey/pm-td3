import numpy as np
from data_loader import DataLoader

class Environment:
    '''### Stock Trading Environment for DDPG agent
    The observation is the feature vector for each stock in the dataset.
    The action is a list of floats, each float representing the fraction of the portfolio to invest in a particular stock
    Args:
        features: (dict) A dictionary containing dataframes for each stock in the dataset
        initial_investment: (float) The initial balance for the agent
    '''
    def __init__(self, data, features, symbols, initial_investment):
        self.data = data
        self.features = features
        self.symbols = symbols
        self.initial_investment = initial_investment

        # Trading fees
        self.p_bid_fee = 0.01                               # Buying fee (+1% to the price)
        self.p_ask_fee = 0.005                              # Selling fee (-0.5% from price)
        
        # Initialize the environment
        self.t_curr = None                                  # Current time step
        self.t_end = len(self.data[symbols[0]])             # End time step
        self.cash = initial_investment                      # Cash balance
        self.portfolio = {symbol: 0 for symbol in symbols}  # Portfolio positions
        self.total_value = initial_investment               # Portfolio value
        self.history = []                                   # Trading history

    #----------------------------------------
    def reset(self):
        '''### Reset the environment
        Returns:
            obs (np.array): The initial observation
        '''
        self.t_curr = 0
        self.cash = self.initial_investment
        self.portfolio = {symbol: 0 for symbol in self.symbols}
        self.total_value = self.initial_investment
        self.history = []
        return self._get_state()
    
    #----------------------------------------
    def step(self, actions):
        '''### Take a step in the environment
        Args:
            actions (np.array): The actions to take
        Returns:
            next_state (np.array): The next state
            reward (float): The reward for the actions
            done (bool): Whether the episode is complete
        '''
        self._rebalance_portfolio(actions)      # Rebalance the portfolio
        self.t_curr += 1                        # Move to the next time step
        reward = self._compute_reward()         # Compute the reward
        done = self.t_curr >= self.t_end - 1    # Check if the episode is complete
        next_state = self._get_state()          # Get the next state

        # Save the trading history
        self.history.append({
            'cash': self.cash,
            'portfolio': self.portfolio.copy(),
            'value': self.total_value
        })

        return next_state, reward, done, {}
    
    #----------------------------------------
    def _rebalance_portfolio(self, target_allocations):
        '''### Rebalance the portfolio to the target allocations
        Args:
            target_allocations (np.array): The target allocations for each stock
        '''
        # Get portfolio and stock prices
        curr_shares = np.array([self.portfolio[symbol] for symbol in self.symbols])
        curr_prices = np.array([self.data[symbol].iloc[self.t_curr]['Close'] for symbol in self.symbols])

        curr_values = curr_shares * curr_prices
        total_value = self.cash + np.sum(curr_values)

        target_values = target_allocations * total_value

        target_shares = target_values / curr_prices
        shares_to_trade = target_shares - curr_shares

        buy_shares = np.floor(np.maximum(0, shares_to_trade))
        sell_shares = np.floor(np.maximum(0, -shares_to_trade))
        total_shares = (curr_shares + buy_shares - sell_shares).astype(int)

        buy_costs = buy_shares * curr_prices * (1 + self.p_bid_fee)
        sell_proceeds = sell_shares * curr_prices * (1 - self.p_ask_fee)

        total_costs = np.sum(buy_costs)
        total_proceeds = np.sum(sell_proceeds)

        self.cash = self.cash - total_costs + total_proceeds
        self.portfolio = {symbol: shares for symbol, shares in zip(self.symbols, total_shares)}
        self.total_value = self.cash + np.sum(total_shares * curr_prices)

    #----------------------------------------
    def _compute_reward(self):
        '''### Compute the reward for the current step
        Returns:
            reward (float): The reward
        '''
        reward = (self.total_value - self.initial_investment)
        if self.cash < 0:
            reward -= self.cash
        reward = reward * 100 / self.initial_investment
        return reward
    
    #----------------------------------------
    def _get_state(self):
        '''### Get the current environment state
        Returns:
            state (np.array): The current state
        '''
        state = []
        for symbol in self.symbols:
            state.append(self.features[symbol].iloc[self.t_curr].values)
        state = np.array(state)
        return state
