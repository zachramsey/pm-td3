import numpy as np
from collections import deque


class TradingEnv:
    '''### Trading Environment for Portfolio Management
    Args:
        data (dict): The stock price data
        features (dict): The stock features
        initial_investment (int): The initial investment
        buy_fee (float): The buying fee
        sell_fee (float): The selling fee
        window_size (int): The window size for the state
    '''
    def __init__(self, data, features, initial_investment, buy_fee, sell_fee, window_size):
        self.data = data
        self.features = features
        self.initial_investment = initial_investment
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.window_size = window_size

        self.stock_symbols = list(data.keys())
        self.dates = self.data[self.stock_symbols[0]].index
        self.current_step = 0

        self.portfolio_value = initial_investment
        self.cash = initial_investment
        self.holdings = {symbol: 0 for symbol in self.stock_symbols}

        self.state_queue = deque(maxlen=window_size)

    #--------------------------------
    def reset(self):
        '''### Reset the environment to the initial state
        Returns:
            np.array: The initial state
        '''
        self.current_step = 0
        self.cash = self.initial_investment
        self.holdings = {symbol: 0 for symbol in self.stock_symbols}
        self.portfolio_value = self.initial_investment

        initial_state = self._get_state()
        for _ in range(self.window_size):
            self.state_queue.append(initial_state)

        return self._get_stacked_state()

    #--------------------------------
    def step(self, actions):
        '''### Take a step in the environment
        Args:
            actions (list): The list of actions for each stock
        Returns:
            np.array: The next state
            float: The reward for the step
            bool: The termination signal
        '''
        self._execute_trades(actions)
        self.current_step += 1

        next_state = self._get_state()
        self.state_queue.append(next_state)
        stacked_next_state = self._get_stacked_state()

        reward = self._get_portfolio_value() - self.portfolio_value
        self.portfolio_value += reward

        done = self.current_step >= len(self.dates) - 1

        return stacked_next_state, reward, done

    #--------------------------------
    def _get_state(self):
        '''### Get the current state of the environment
        Returns:
            np.array: The current state
        '''
        state = []
        date = self.dates[self.current_step]
        for symbol in self.stock_symbols:
            state.extend(self.features[symbol].loc[date].values)
            state.append(self.holdings[symbol])
        state.append(self.cash)
        return np.array(state)

    #--------------------------------
    def _get_stacked_state(self):
        '''### Get the stacked state of the environment
        Returns:
            np.array: The stacked state
        '''
        return np.concatenate(self.state_queue, axis=0)

    #--------------------------------
    def _execute_trades(self, actions):
        '''### Execute the trades based on the actions
        Args:
            actions (list): The list of actions for each stock
        '''
        for i, symbol in enumerate(self.stock_symbols):
            action = actions[i]
            if action > 0:
                self._buy_stock(symbol, action)
            elif action < 0:
                self._sell_stock(symbol, -action)

    #--------------------------------
    def _buy_stock(self, symbol, amount):
        '''### Buy a stock
        Args:
            symbol (str): The stock symbol
            amount (float): The amount of stock to buy
        '''
        price = self.data[symbol].iloc[self.current_step]['close']
        total_cost = price * amount * (1 + self.buy_fee)
        if total_cost <= self.cash:
            self.cash -= total_cost
            self.holdings[symbol] += amount

    #--------------------------------
    def _sell_stock(self, symbol, amount):
        '''### Sell a stock
        Args:
            symbol (str): The stock symbol
            amount (float): The amount of stock to sell
        '''
        if self.holdings[symbol] >= amount:
            price = self.data[symbol].iloc[self.current_step]['close']
            total_return = price * amount * (1 - self.sell_fee)
            self.cash += total_return
            self.holdings[symbol] -= amount

    #--------------------------------
    def _get_portfolio_value(self):
        '''### Get the total value of the portfolio
        Returns:
            float: The total value of the portfolio
        '''
        total_value = self.cash
        for symbol in self.stock_symbols:
            price = self.data[symbol].iloc[self.current_step]['close']
            total_value += price * self.holdings[symbol]
        return total_value
