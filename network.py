import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F

def network(layer_sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes)-1):
        act = activation if i < len(layer_sizes)-2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), act()]
    return nn.Sequential(*layers)

class DeepQNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, activation):
        super().__init__()
        
        self.q = network([state_shape[0]+action_shape[0]]+list(hidden_sizes)+[1], activation)


    def forward(self, state, action):
        """Forward pass

        Args:
            state (ndarray): state of dynamics
            action (ndarray): action appled to the env.

        Returns:
            q values: q values for each action given the state
        """        
        
        q = self.q(torch.cat([state, action], dim=-1))
        q = torch.squeeze(q, -1)

        return q

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, activation, action_limit):
        super().__init__()
        self.net = network([state_shape[0]] + list(hidden_sizes), activation, activation)
        self.mu = nn.Linear(hidden_sizes[-1], action_shape[0])
        self.log_std = nn.Linear(hidden_sizes[-1], action_shape[0])
        self.action_limit = action_limit

    def forward(self, state, deterministic=False, with_logprob=True):
        """Forward pass

        Args:
            state (ndarray): state of dynamics
            deterministic (bool, optional): only true when evaluation. Defaults to False.
            with_logprob (bool, optional): compute log probability for pi. Defaults to True.

        Returns:
            action: action outputting by the parametrized policy
            logp_pi: log probability for pi
        """        
        net_out = self.net(state)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        distribution = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = distribution.rsample()

        if with_logprob:
            # see appendix C of SAC paper; this is more numerically-stable version of Eq 21.
            logp_pi = distribution.log_prob(action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp_pi = None

        action = torch.tanh(action)
        action = self.action_limit * action

        return action, logp_pi

class ActorCritic(nn.Module):

    def __init__(self, state_shape, action_shape, action_limit, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # policy
        self.pi = SquashedGaussianNetwork(state_shape, action_shape, hidden_sizes, activation, action_limit)
        # q value functions
        self.q1 = DeepQNetwork(state_shape, action_shape, hidden_sizes, activation)
        self.q2 = DeepQNetwork(state_shape, action_shape, hidden_sizes, activation)

    def act(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.pi(state, deterministic, with_logprob=False)
            return action.numpy()
