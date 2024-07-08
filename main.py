import torch
from collections import deque

from data_loader import DataLoader
from trading_env import TradingEnv
from td3 import TD3

#================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # Load the data and features dictionaries
    dl = DataLoader(symbols)
    data = dl.get_data(symbols)          # {Symbol: Dataframe}
    features = dl.get_features(symbols)  # {Symbol: Dataframe}

    # 90/10 train-test split
    split = int(0.9 * len(data[symbols[0]]))
    data_train = {symbol: data[symbol].iloc[:split] for symbol in symbols}
    data_test = {symbol: data[symbol].iloc[split:] for symbol in symbols}
    feat_train = {symbol: features[symbol].iloc[:split] for symbol in symbols}
    feat_test = {symbol: features[symbol].iloc[split:] for symbol in symbols}

    # Initialize the environment
    initial_investment = 100000
    buy_fee = 0.001
    sell_fee = 0.001
    window_size = 4

    env = TradingEnv(data_train, feat_train, initial_investment, buy_fee, sell_fee, window_size)

    # Initialize the agent
    n_stocks = len(features)
    n_features = len(features[symbols[0]].columns)
    conv_out = 64
    n_heads = 4
    n_hidden = 128
    actor_lr = 1e-3
    critic_lr = 1e-3
    max_action = 1.0
    buffer_size = int(1e6)
    alpha = 0.6
    beta = 0.4
    beta_increment = 0.001
    frame_stack = 4
    n_step = 3
    gamma = 0.99
    tau = 0.005

    agent = TD3(n_stocks, n_features, conv_out, n_heads, n_hidden, 
                actor_lr, critic_lr, max_action, buffer_size, alpha, 
                beta, beta_increment, gamma, tau, frame_stack, n_step)

    # Train the agent
    n_episodes = 100
    batch_size = 100
    state_queue = deque(maxlen=frame_stack)

    for episode in range(n_episodes):
        state = env.reset()
        state_queue.extend([state] * frame_stack)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state_queue)
            next_state, reward, done = env.step(action)
            state_queue.append(next_state)
            agent.add_experience(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.train(batch_size)

        print(f'Episode: {episode + 1}, Reward: {episode_reward:.2f}, env.portfolio_value: {env._get_portfolio_value():.2f}')

    # Test the agent
    state = env.reset()
    state_queue.extend([state] * frame_stack)
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state_queue)
        next_state, reward, done = env.step(action)
        state_queue.append(next_state)
        episode_reward += reward

    print(f'Test Reward: {episode_reward:.2f}')