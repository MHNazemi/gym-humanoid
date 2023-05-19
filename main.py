from __future__ import annotations
import atexit
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import gymnasium as gym
from agents import *
import argparse
torch.autograd.set_detect_anomaly(True)
plt.rcParams["figure.figsize"] = (10, 5)


if __name__=="__main__":
    # Create and wrap the environment
    env_name = "Humanoid-v4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='?', const=1, type=int,default=1)
    parser.add_argument('--load', nargs='?', const=1, type=int,default=0)
    parser.add_argument('--display', nargs='?', const=1, type=int,default=0)
    parser.add_argument('--save', nargs='?', const=1, type=int,default=1)
    parser.add_argument('--agent', nargs='?', type=str,default="ac",choices=["r","ac"])
    
    args = parser.parse_args()

    if args.display==1:
        env = gym.make(env_name,render_mode="human" )
    else:
        env = gym.make(env_name)

        
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(1e4)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    for seed in [1]: # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        if args.agent=="r":
            agent = REINFORCE(obs_space_dims, action_space_dims)
        elif args.agent=="ac":
            agent = Actor_Critic(obs_space_dims, action_space_dims)


        def cleanUp():
            if args.save==1:
                agent.saveModel()
            rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
            df1 = pd.DataFrame(rewards_to_plot).melt()
            df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
            sns.set(style="darkgrid", context="talk", palette="rainbow")
            sns.lineplot(x="episodes", y="reward", data=df1).set(
                title="REINFORCE for InvertedPendulum-v4"
            )
            # plt.show()
        atexit.register(cleanUp)

        if args.load==1:
            agent.loadModel()

        reward_over_episodes = []





        # state, info = wrapped_env.reset(seed=seed)
        # agent.test(state,state)

        # action = agent.sample_action(state)
        # state_new, reward_new, terminated, truncated, info = wrapped_env.step(action)
        # action_new = agent.sample_action(state)
        # agent.update(reward_new,state,state_new,action,action_new)

        # action_new = agent.sample_action(state)
        # action=action_new
        # state_new, reward_new, terminated, truncated, info = wrapped_env.step(action)
        # action_new = agent.sample_action(state)
        # agent.update(reward_new,state,state_new,action,action_new)

        

        
        
        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            state, info = wrapped_env.reset(seed=seed)
            done = False
            try:
                while not done:
                    if args.display==1:
                        time.sleep(1/30)


                    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                    # These represent the next observation, the reward from the step,
                    # if the episode is terminated, if the episode is truncated and
                    # additional info from the step

                    if args.agent=="r":
                        action = agent.sample_action(state)
                        state, reward, terminated, truncated, info = wrapped_env.step(action)
                        agent.rewards.append(reward)

                    elif  args.agent=="ac" :

                        action,log_prob = agent.sample_action(state) # action based on updated model
                        state_new, reward, terminated, truncated, info = wrapped_env.step(action)

                        if args.train==1:
                            action_new,_ = agent.sample_action(state_new) # fake action 
                            agent.update(reward,state,state_new,action,log_prob,action_new) # train
                            
                        state=state_new

                        

                    
                        
                    

                    # End the episode when either truncated or terminated is true
                    #  - truncated: The episode duration reaches max number of timesteps
                    #  - terminated: Any of the state space values is no longer finite.
                    done = terminated or truncated



                reward_over_episodes.append(wrapped_env.return_queue[-1])
                if  args.agent=="r" and args.train==1:
                    agent.update()

                if episode % 1 == 0:
                    avg_reward = int(np.mean(wrapped_env.return_queue))
                    print("Episode:", episode, "Average Reward:", avg_reward)

                if episode % 100 == 0 and args.save ==1:
                    agent.saveModel()  
            except Exception as ex:
                print(ex)
                break
        rewards_over_seeds.append(reward_over_episodes)
        if args.save==1:
            agent.saveModel()  
    

    