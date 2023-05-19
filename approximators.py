"""
The Policy NN atchitecture is inspired by: 
https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym


class Policy_Network(nn.Module):
    """
    Policy Network
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initilizes a NN as an policy function. 
        It uses Gaussian distribution to sample the actions

        Args:
            state_dim (int): dimension of observable features of a state
            action_dim (int): dimension of actions
        """

        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)

        hidden_space1 = 256  
        hidden_space2 = 128  
        hidden_space3 = 32  
    
        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space3),
            nn.ReLU()
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space3, action_dim)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space3, action_dim)
        )
    
    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ inputs observation into the NN and returns means and standard deviations of the Gaussian distribution based on observation
        Args:
            observation: state
        Returns:
            action_means: predicted mean 
            action_stddevs: predicted standard deviation 
        """

        shared_features = self.shared_net(observation.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs


class Action_State_Network(nn.Module):

    """
    Value Network
    """

    def __init__(self, state_dim: int, action_dim: int):
        """
        Initilizes a NN as an value function. 

        Args:
            state_dim (int): dimension of observable features of a state
            action_dim (int): dimension of actions
        """

        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        
        hidden_space1 = 256
        hidden_space2 = 64 
    
        # Q(s,a) value
        self.Q = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2,action_dim)
        )

#         state_embedding_network = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embedding_dim)
# )       
#         action_embedding_network = nn.Sequential(
#             nn.Linear(action_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embedding_dim)
#         )

#         merged_input = torch.cat((state_embedding, action_embedding), dim=1)
#         merged_input = torch.cat((state_embedding, action_embedding), dim=1)

#         q_network = nn.Sequential(
#             nn.Linear(embedding_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, q_value_dim)
#         )
        
      

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ inputs observation into the NN and returns the value of the state/action 
        Args:
            observation: state + action 
        Returns:
            action_means: predicted mean 
            action_stddevs: predicted standard deviation 
        """
        
        value = self.Q(observation.float())

        return value
    