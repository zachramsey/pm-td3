import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import PrioritizedReplay

#================================================================
class Actor(nn.Module):
    def __init__(self, n_features, conv_out=64, n_heads=4, n_hidden=128):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_out, kernel_size=1)
        self.attn_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=n_heads, dim_feedforward=n_hidden, batch_first=True)
        self.state_encoder = nn.TransformerEncoder(self.attn_layer, num_layers=1)
        self.fc1 = nn.Linear(conv_out, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # Initialize the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    #--------------------------------
    def forward(self, state):
        # State shape: (batches, n_stocks, n_features)
        x = state.permute(0, 2, 1)      # Permute to shape:     (batches, n_features, n_stocks)
        x = self.conv1(x)               # Apply convolution:    (batches, conv_out, n_stocks)
        
        x = x.permute(0, 2, 1)          # Permute to shape:     (batches, n_stocks, conv_out)
        x = self.state_encoder(x)       # Apply self-attn:      (batches, n_stocks, conv_out)

        x = F.relu(self.fc1(x))         # Apply fc1:            (batches, n_stocks, n_hidden)
        x = self.fc2(x)                 # Apply fc2:            (batches, n_stocks, 1)

        x = x.squeeze(-1)               # Remove last dim:      (batches, n_stocks)
        x = F.softmax(x, dim=-1)        # Apply softmax:        (batches, n_stocks)
        return x

#================================================================
class Critic(nn.Module):
    def __init__(self, n_stocks, n_features, conv_out, n_heads, n_hidden):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_out, kernel_size=1)
        self.state_attn_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=n_heads, dim_feedforward=n_hidden, batch_first=True)
        self.state_encoder = nn.TransformerEncoder(self.state_attn_layer, num_layers=1)
        self.action_fc = nn.Linear(n_stocks, conv_out)
        self.cross_attn = nn.MultiheadAttention(embed_dim=conv_out, num_heads=n_heads, kdim=conv_out, vdim=conv_out, batch_first=True)
        self.fc1 = nn.Linear(conv_out, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # Initialize the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    #--------------------------------
    def forward(self, s, a):
        # State shape: (batches, n_stocks, n_features)
        # Action shape: (batches, n_stocks)
        x = s.permute(0, 2, 1)          # Permute to shape:     (batches, n_features, n_stocks)
        x = self.conv1(x)               # Apply convolution:    (batches, conv_out, n_stocks)

        x = x.permute(0, 2, 1)          # Permute to shape:     (batches, n_stocks, conv_out)
        x = self.state_encoder(x)       # Apply self-attn:      (batches, n_stocks, conv_out)

        a = a.unsqueeze(1)              # Add a dim:            (batches, 1, n_stocks)
        a = self.action_fc(a)           # Apply fc:             (batches, 1, conv_out)

        x, _ = self.cross_attn(x, a, a) # Apply cross-attn:     (batches, n_stocks, conv_out)
        x = F.relu(self.fc1(x))         # Apply fc1:            (batches, n_stocks, n_hidden)
        x = self.fc2(x)                 # Apply fc2:            (batches, n_stocks, 1)
        x = x.squeeze(-1)               # Remove last dim:      (batches, n_stocks)
        return x

#================================================================
class DDPG:
    '''### Deep Deterministic Policy Gradient (DDPG) Agent
    Args:
        n_stocks (int): The number of stocks in the dataset
        n_features (int): The number of features for each stock
        device (torch.device): The device to run the agent on
        conv_out (int, optional): The number of output channels from the convolutional layer
        n_heads (int, optional): The number of heads for the multihead attention layer
        n_hidden (int, optional): The number of hidden units for the fully connected layers
        actor_lr (float, optional): The learning rate for the actor network
        critic_lr (float, optional): The learning rate for the critic network
        gamma (float, optional): The discount factor
        tau (float, optional): The soft update factor
        buffer_capacity (int, optional): The maximum capacity of the replay buffer
        batch_size (int, optional): The batch size for training
        policy_update_freq (int, optional): The frequency of updating the policy network
    '''
    def __init__(self, n_stocks, n_features, device,
                 conv_out=64, n_heads=4, n_hidden=128,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001, 
                 buffer_capacity=100000, batch_size=32, policy_update_freq=2):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_freq = policy_update_freq
        
        self.actor = Actor(n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.target_actor = Actor(n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.critic1 = Critic(n_stocks, n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.critic2 = Critic(n_stocks, n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.target_critic1 = Critic(n_stocks, n_features, conv_out, n_heads, n_hidden).to(self.device)
        self.target_critic2 = Critic(n_stocks, n_features, conv_out, n_heads, n_hidden).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.replay_buffer = PrioritizedReplay(buffer_capacity)
        
        self._update_target_networks(1.0)
        self.total_it = 0
    
    #--------------------------------
    def _update_target_networks(self, tau):
        '''### Soft update the target networks
        Args:
            tau (float): Soft update factor
        '''
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    #--------------------------------
    def action(self, state):
        '''### Select an action given the state
        Args:
            state (np.array): The current state
        Returns:
            np.array: The action to take
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        return action
    
    #--------------------------------
    def step(self):
        '''### Perform a single step of training'''
        self.total_it += 1

        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones, indices, is_weights = self.replay_buffer.sample(self.batch_size, self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Compute the target Q values
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q1 = self.target_critic1(next_states, target_actions)
            target_q2 = self.target_critic2(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Update the critic networks
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss1 = F.mse_loss(q1, target_q, reduction='none')
        critic_loss2 = F.mse_loss(q2, target_q, reduction='none')
        critic_loss = torch.mean(is_weights * (critic_loss1 + critic_loss2))

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update the actor network
        if self.total_it % self.policy_update_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            self._update_target_networks(self.tau)

            # Update priorities
            new_priorities = (critic_loss1 + critic_loss2).detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, new_priorities)

    #--------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        '''### Store a transition in the replay buffer
        Args:
            state (np.array): The current state
            action (np.array): The action taken
            reward (float): The reward received
            next_state (np.array): The next state
            done (bool): Whether the episode is done
        '''
        self.replay_buffer.push(state, action, reward, next_state, done)
