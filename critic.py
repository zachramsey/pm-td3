
import torch
import torch.nn as nn
import torch.nn.functional as F

#================================================================
class Critic(nn.Module):
    def __init__(self, n_stocks, n_features, conv_out, n_heads, n_hidden):
        super(Critic, self).__init__()

        # Q1 network
        self.q1_conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_out, kernel_size=1)
        self.q1_state_attn_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=n_heads, dim_feedforward=n_hidden, batch_first=True)
        self.q1_state_encoder = nn.TransformerEncoder(self.q1_state_attn_layer, num_layers=1)
        self.q1_action_fc = nn.Linear(n_stocks, conv_out)
        self.q1_cross_attn = nn.MultiheadAttention(embed_dim=conv_out, num_heads=n_heads, kdim=conv_out, vdim=conv_out, batch_first=True)
        self.q1_fc1 = nn.Linear(conv_out, n_hidden)
        self.q1_fc2 = nn.Linear(n_hidden, 1)

        # Q2 network
        self.q2_conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_out, kernel_size=1)
        self.q2_state_attn_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=n_heads, dim_feedforward=n_hidden, batch_first=True)
        self.q2_state_encoder = nn.TransformerEncoder(self.q2_state_attn_layer, num_layers=1)
        self.q2_action_fc = nn.Linear(n_stocks, conv_out)
        self.q2_cross_attn = nn.MultiheadAttention(embed_dim=conv_out, num_heads=n_heads, kdim=conv_out, vdim=conv_out, batch_first=True)
        self.q2_fc1 = nn.Linear(conv_out, n_hidden)
        self.q2_fc2 = nn.Linear(n_hidden, 1)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    #--------------------------------
    def forward(self, s, a):
        '''### Forward pass of the critic network
        Args:
            s (torch.Tensor): The input state (batches, n_stocks, n_features)
            a (torch.Tensor): The input action (batches, n_stocks)
        Returns:
            torch.Tensor: The output Q value (batches, 1)
        '''
        # Q1 network
        q1_x = F.relu(self.q1_conv1(s))                     # (batches, n_stocks, conv_out)
        q1_x = self.q1_state_encoder(q1_x)                  # (batches, n_stocks, conv_out)
        q1_a = F.relu(self.q1_action_fc(a.unsqueeze(-1)))   # (batches, n_stocks, conv_out)
        q1_x = q1_x.permute(1, 0, 2)                        # (n_stocks, batches, conv_out)
        q1_a = q1_a.permute(1, 0, 2)                        # (n_stocks, batches, conv_out)
        q1_x, _ = self.q1_cross_attn(q1_a, q1_x, q1_x)      # (n_stocks, batches, conv_out)
        q1_x = q1_x.permute(1, 0, 2)                        # (batches, n_stocks, conv_out)
        q1_x = F.relu(self.q1_fc1(q1_x))                    # (batches, n_stocks, n_hidden)
        q1_x = self.q1_fc2(q1_x)                            # (batches, n_stocks, 1)
        q1_x = q1_x.sum(dim=1, keepdim=True)                # (batches, 1)

        # Q2 network
        q2_x = F.relu(self.q2_conv1(s))                     # (batches, n_stocks, conv_out)
        q2_x = self.q2_state_encoder(q2_x)                  # (batches, n_stocks, conv_out)
        q2_a = F.relu(self.q2_action_fc(a.unsqueeze(-1)))   # (batches, n_stocks, conv_out)
        q2_x = q2_x.permute(1, 0, 2)                        # (n_stocks, batches, conv_out)
        q2_a = q2_a.permute(1, 0, 2)                        # (n_stocks, batches, conv_out)
        q2_x, _ = self.q2_cross_attn(q2_a, q2_x, q2_x)      # (n_stocks, batches, conv_out)
        q2_x = q2_x.permute(1, 0, 2)                        # (batches, n_stocks, conv_out)
        q2_x = F.relu(self.q2_fc1(q2_x))                    # (batches, n_stocks, n_hidden)
        q2_x = self.q2_fc2(q2_x)                            # (batches, n_stocks, 1)
        q2_x = q2_x.sum(dim=1, keepdim=True)                # (batches, 1)

        return q1_x, q2_x

    #--------------------------------
    def q1(self, state, action):
        '''### Forward pass of the Q1 network
        Args:
            state (torch.Tensor): The input state (batches, n_stocks, n_features)
            action (torch.Tensor): The input action (batches, n_stocks)
        Returns:
            torch.Tensor: The output Q value (batches, 1)
        '''
        q1_x = F.relu(self.q1_conv1(state))                     # (batches, n_stocks, conv_out)
        q1_x = self.q1_state_encoder(q1_x)                      # (batches, n_stocks, conv_out)
        q1_a = F.relu(self.q1_action_fc(action.unsqueeze(-1)))  # (batches, n_stocks, conv_out)
        q1_x = q1_x.permute(1, 0, 2)                            # (n_stocks, batches, conv_out)
        q1_a = q1_a.permute(1, 0, 2)                            # (n_stocks, batches, conv_out)
        q1_x, _ = self.q1_cross_attn(q1_a, q1_x, q1_x)          # (n_stocks, batches, conv_out)
        q1_x = q1_x.permute(1, 0, 2)                            # (batches, n_stocks, conv_out)
        q1_x = F.relu(self.q1_fc1(q1_x))                        # (batches, n_stocks, n_hidden)
        q1_x = self.q1_fc2(q1_x)                                # (batches, n_stocks, 1)
        q1_x = q1_x.sum(dim=1, keepdim=True)                    # (batches, 1)
        return q1_x
    