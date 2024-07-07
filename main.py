import numpy as np
import torch

from data_loader import DataLoader
from environment import Environment
from ddpg import DDPG

#----------------------------------------
def train(env, agent, num_episodes, max_timesteps):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.step()
            state = next_state
            episode_reward += reward
            
            if done:
                break

            if t % 50 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Timestep {t}, Reward: {reward}, Value: {env.total_value}, Cash: {env.cash}          ", end='\r')
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Value: {env.total_value}, Cash: {env.cash}                                  ")

#----------------------------------------
def test(env, agent, num_episodes, max_timesteps):
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Value: {env.total_value}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} test episodes: {avg_reward}, Final Portfolio Value: {env.total_value}")
        
#================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    initial_investment = 25000
    n_stocks = len(symbols)

    conv_out = 256
    n_heads = 8
    n_hidden = 512

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

    # Number of stocks and features
    n_stocks = len(features)
    n_features = len(features[symbols[0]].columns)

    # Training
    env = Environment(data_train, feat_train, symbols, initial_investment)
    agent = DDPG(n_stocks, n_features, device)
    max_episodes = 100
    max_timesteps = 10000
    train(env, agent, max_episodes, max_timesteps)

    # Testing
    env_test = Environment(data_test, feat_test, symbols, initial_investment)
    num_episodes = 10
    max_timesteps = 200
    test(env_test, agent, num_episodes, max_timesteps)