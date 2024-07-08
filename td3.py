import torch
import torch.nn as nn
import torch.optim as optim

import copy
from collections import deque

from actor import Actor
from critic import Critic
from exp_replay import PrioritizedReplayBuffer

#================================================================
class TD3:
    '''### Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent'''
    def __init__(self, n_stocks, n_features, 
                 conv_out, n_heads, n_hidden, actor_lr, critic_lr,
                 max_action, buffer_size, alpha, beta, beta_increment, 
                 gamma, tau, frame_stack, n_step):
        
        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the actor and critic networks
        self.actor = Actor(n_features, max_action, conv_out, n_heads, n_hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(n_stocks, n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)

        # Set the hyperparameters
        self.max_action = max_action            # Maximum action value
        self.beta = beta                        # Importance sampling weight
        self.beta_increment = beta_increment    # Increment for importance sampling weight

        self.discount = gamma                   # Discount factor
        self.tau = tau                          # Target network update rate
        self.policy_noise = 0.2                 # Noise added to target policy during critic update
        self.noise_clip = 0.5                   # Range to clip target policy noise
        self.policy_freq = 2                    # Delayed policy update frequency
        self.total_it = 0                       # Total iterations

        self.frame_stack = frame_stack          # Number of frames to stack
        self.n_step = n_step                    # Number of steps for n-step return
        self.n_step_buffer = deque(maxlen=n_step)   # Buffer for n-step return

    #--------------------------------    
    def select_action(self, state):
        '''### Select an action from the policy
        Args:
            state (np.array): The input state (n_stocks, n_features)
        Returns:
            np.array: The selected action (n_stocks)
        '''
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    #--------------------------------
    def add_experience(self, state, action, reward, next_state, done):
        '''### Add a new transition to the replay buffer
        Args:
            state (np.array): The current state (n_stocks, n_features)
            action (np.array): The action taken (n_stocks)
            reward (float): The reward received
            next_state (np.array): The next state (n_stocks, n_features)
            done (bool): The termination signal
        '''
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_state, done = self.n_step_return()
        state, action = self.n_step_buffer[0][:2]
        self.replay_buffer.add(reward, (state, next_state, action, reward, done))

    #--------------------------------
    def n_step_return(self):
        '''### Compute the n-step return
        Returns:
            float: The n-step return
            np.array: The next state (n_stocks, n_features)
            bool: The termination signal
        '''
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, next_s, d = transition[-3:]
            reward = r + self.discount * reward * (1 - d)
            next_state, done = (next_s, d) if d else (next_state, done)
        return reward, next_state, done

    #--------------------------------
    def train(self, batch_size=100):
        '''### Train the agent
        Args:
            batch_size (int): The batch size for training
        '''
        self.total_it += 1

        # Sample a batch of transitions from the replay buffer
        batch, indices, sampling_probabilities = self.replay_buffer.sample(batch_size)

        state, next_state, action, reward, done = zip(*batch)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update priorities in the replay buffer
        errors = torch.abs(current_Q1 - target_Q).detach().cpu().numpy()
        for i in range(batch_size):
            self.replay_buffer.update(indices[i], errors[i])

        # Update beta parameter for prioritized experience replay
        self.beta = min(1.0, self.beta + self.beta_increment)