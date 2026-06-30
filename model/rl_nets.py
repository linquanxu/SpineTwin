import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):


    def __init__(self, action_dim, hidden_dim=256, is_3d=True):
        super().__init__()
        self.patch_size = 32       
        in_channels = 4  # image + heatmap + grad_y + grad_x
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), 
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.patch_size + 1, self.patch_size + 1)
            dummy_out = self.feature_extractor(dummy)
            conv_out_dim = dummy_out.shape[1]

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        nn.init.constant_(self.value_stream[-1].bias, 0.0)
        nn.init.constant_(self.advantage_stream[-1].bias, 0.0)

    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values