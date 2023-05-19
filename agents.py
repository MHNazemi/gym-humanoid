from __future__ import annotations
import numpy as np
import torch
from torch.distributions.normal import Normal
from approximators import *

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)
        
        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss = log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

    def saveModel(self):
        print("save model")
        torch.save(self.net.state_dict(),"model2.pt")

    def loadModel(self):
        self.net.load_state_dict(torch.load("Models/model2.pt"))
        self.net.eval()



class Actor_Critic:
    """Actor Critic using TD(0) algorithm."""
    
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate_valuef = 1e-4  # Learning rate for policy optimization
        self.learning_rate_policy = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability
        self.i=0

        self.actions = []

        self.policy = Policy_Network(obs_space_dims, action_space_dims)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate_policy)


        self.value = Action_State_Network(obs_space_dims, action_space_dims)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.learning_rate_valuef)

        self.is_first_clone_skipped=False

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.policy(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        
        # self.actions.append((distrib,action))
        log_prob = distrib.log_prob(action)
        # self.log_probs.append(prob)


        action = action.numpy()


        return action,log_prob
    
    def update(self,r,s,s_new,a,log_prob_a,a_new):
        """Updates the policy network's weights."""
        """Updates the value network's weights."""

        
        # updates value function


        # ─── Conversion Of Numpy Arrays To Tensors ────────────────────
        s =torch.tensor(s)
        s_new =torch.tensor(s_new)


        a =torch.tensor(a)
        a_new =torch.tensor(a_new)
        # ──────────────────────────────────────────────────────────────

        

        # ─── Normalize States ─────────────────────────────────────────
      
        # Compute the min and max values of the states
        s_min = torch.min(s, dim=0).values
        s_max = torch.max(s, dim=0).values

        # Normalize the states to the range [-1, 1]
        normalized_s= 2 * (s - s_min) / (s_max - s_min) - 1


        s_min = torch.min(s_new, dim=0).values
        s_max = torch.max(s_new, dim=0).values

        # Normalize the states to the range [-1, 1]
        normalized_s_new= 2 * (s_new - s_min) / (s_max - s_min) - 1
        # ──────────────────────────────────────────────────────────────

        


        Q_s_a = torch.cat((normalized_s,a))
        Q_snew_anew = torch.cat((normalized_s_new,a_new))

        value_s_a = self.value(Q_s_a)
        value_snew_anew = self.value(Q_snew_anew)

        # with torch.no_grad():
        #     value_snew_anew = self.value(Q_snew_anew)

       


        # # # updates policy netowrk 
        
        loss = 0
        loss = (log_prob_a* value_s_a.detach() ).mean() * (-1)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        print (f"policy loss: {loss} ")

        # print(f"\npolicy grad {se}")

        # # # updates value network
        loss=0
        loss = r+ self.gamma*value_snew_anew.mean() - value_s_a.detach().mean()
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()


        


        

        # running_g = 0
        # gs = []

        # # Discounted return (backwards) - [::-1] will return an array in reverse
        # for R in self.rewards[::-1]:
        #     running_g = R + self.gamma * running_g
        #     gs.insert(0, running_g)

        # deltas = torch.tensor(gs)

        # loss = 0
        # # minimize -1 * prob * reward obtained
        # for log_prob, delta in zip(self.probs, deltas):
        #     loss += log_prob.mean() * delta * (-1)
        
        # # Update the policy network
        # self.policy_optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # Empty / zero out all episode-centric/related variables
        # self.probs = []
        # self.rewards = []

    def saveModel(self):
        print("save model")
        torch.save(self.policy.state_dict(),f"Models/model_{self.i}_policy.pt")
        torch.save(self.value.state_dict(),f"Models/model_{self.i}_value.pt")
        self.i +=1

    def loadModel(self):
        self.i = 10
        self.policy.load_state_dict(torch.load(f"Models/model_{self.i}_policy.pt"))
        self.policy.eval()
        self.value.load_state_dict(torch.load(f"Models/model_{self.i}_value.pt"))
        self.value.eval()

    def test(self,input1,input2):
        input1 = torch.tensor(np.array([input1]))
        input2 = torch.tensor(np.array([input2]))

        action_means, action_stddevs = self.policy(input1)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        log_prob1 = distrib.log_prob(action)

        # Fetch output2 before backward pass for loss1
        action_means, action_stddevs = self.policy(input2)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        log_prob2 = distrib.log_prob(action)

        # Backpropagation and parameter update for the first output
        loss1 = log_prob1.mean() * float("5") * (-1)
        self.policy_optimizer.zero_grad()
        loss1.backward()
        self.policy_optimizer.step()

      

        # Forward pass for the second output
        loss2 = log_prob2.mean() * float("5") * (-1)
        # Backpropagation and parameter update for the second output
        self.policy_optimizer.zero_grad()
        loss2.backward()
        self.policy_optimizer.step()