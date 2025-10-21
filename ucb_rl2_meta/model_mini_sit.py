# ucb_rl2_meta/model_mini_sit.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from ucb_rl2_meta.mini_sit import MiniSiT_1GSA

class Policy_MiniSiT(nn.Module):
    def __init__(self, obs_shape, action_dim, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.recurrent_hidden_state_size = 1  # feedforward (required by RolloutStorage)

        c, h, w = obs_shape
        self.encoder = MiniSiT_1GSA(
            img_size=h,
            in_chans=c,
            embed_dim=64,
            num_heads=4,
            depth=2,
            patch=8
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        return self.encoder(obs)

    def act(self, obs, rnn_hxs, masks, deterministic=False):
        features = self.encoder(obs)                       # (B, 128)
        logits   = self.actor(features)                    # (B, A)
        dist     = Categorical(logits=logits)

        if deterministic:
            action_idx = dist.probs.argmax(dim=-1)         # (B,)
        else:
            action_idx = dist.sample()                     # (B,)

        action_log_prob = dist.log_prob(action_idx).unsqueeze(-1)  # (B,1)
        value           = self.critic(features)                     # (B,1)

        # RolloutStorage expects Discrete actions as indices with shape (B,1), dtype long
        actions = action_idx.unsqueeze(-1).long()          # (B,1)

        return value, actions, action_log_prob, rnn_hxs

    def get_value(self, obs, rnn_hxs, masks):
        features = self.encoder(obs)
        return self.critic(features)
    
    def evaluate_actions(self, obs, rnn_hxs, masks, actions):
        """
        Evaluate policy given observations and taken actions.
        Used by PPO to compute loss terms (value, log_probs, entropy).
        """
        features = self.encoder(obs)
        logits   = self.actor(features)
        dist     = torch.distributions.Categorical(logits=logits)

        # actions come in shape [B,1], squeeze for Categorical
        actions = actions.squeeze(-1)
        action_log_probs = dist.log_prob(actions).unsqueeze(-1)
        dist_entropy     = dist.entropy().mean()
        values           = self.critic(features)

        return values, action_log_probs, dist_entropy

