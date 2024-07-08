
import torch
import torch.nn as nn
import torch.nn.functional as F

#================================================================
class Actor(nn.Module):
    def __init__(self, n_features, max_action, conv_out=64, n_heads=4, n_hidden=128):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_out, kernel_size=1)
        self.attn_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=n_heads, dim_feedforward=n_hidden, batch_first=True)
        self.state_encoder = nn.TransformerEncoder(self.attn_layer, num_layers=1)
        self.fc1 = nn.Linear(conv_out, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        self.max_action = max_action

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    #--------------------------------
    def forward(self, state):
        '''### Forward pass of the actor network
        Args:
            state (torch.Tensor): The input state (batches, n_stocks, n_features)
        Returns:
            torch.Tensor: The output action (batches, n_stocks)
        '''
        x = F.relu(self.conv1(state))           # (batches, n_stocks, conv_out)
        x = self.state_encoder(x)               # (batches, n_stocks, conv_out)
        x = F.relu(self.fc1(x))                 # (batches, n_stocks, n_hidden)
        x = self.fc2(x)                         # (batches, n_stocks, 1)
        x = torch.tanh(x) * self.max_action     # (batches, n_stocks, 1)
        return x.squeeze(-1)                    # (batches, n_stocks)
