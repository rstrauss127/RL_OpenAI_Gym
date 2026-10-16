# Q-learning algorithm for FrozenLake-v1 using Epsilson Greedy Policy

import gymnasium  as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
def run(episodes=10000, is_training=True, render=False):
    # Create environment - 8x8 grid Slippery FrozenLakev1
    env = gym.make("FrozenLake-v1", map_name='8x8', is_slippery=True, render_mode= 'human' if render else None)
    # is_slippery=True means there is a chance of the agent 'slipping' on the ice and moving in a different direction than intended. 
    # This addes a level of uncertainty to the environment, making traditional path-finding algorithms uneffective. 
    # Note: If the gaent tries to move off the grid, it will stay in the same position.
    
    if is_training:
        # Create empty Q-table for 64x4 
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else: # load Q-table from file
        f = open(os.path.join("QTable","FrozenLake_Qtable.pkl"), "rb")
        q = pickle.load(f)
        f.close()

    # Parameters
    alpha_learning_rate = 0.8 # Values closed to 1 mean the agent will learn faster, but it may not converge to the optimal policy. Values close to 0 mean the agent will learn slower, but it may converge to the optimal policy.
    gamma_discount_factor = 0.9 # Values closed to 1 mean the agent will prioritize future rewards, while values close to 0 mean the agent will prioritize immediate rewards.
    # Epislion Greedy Policy Parameters
    epsilon = 1 # Starts at at 1(100% random actions) and overtime we will decrease the randomness using the decay rate.
    epsilon_decay_rate = 0.0001 # At 0.0001, the agent will need 10,00 episodes to reach 0.
    random_generator = np.random.default_rng(0)

    rewards_per_episode = np.zeros(episodes) # Storage for rewards for each episode

    # Start training loop
    for i in range(episodes):
        # Get state: For 8x8 grid, the state is a number between 0 and 63, representing the position in the grid
        state = env.reset()[0]
        terminated = False # True if the episode has ended: the agent has reached the goal or fallen into a hole
        truncated = False # True when action limit is reached

        while(not terminated and not truncated):
        # Epsilon Greedy Policy: Choose a random action with probability epsilon    
            if is_training and random_generator.random() < epsilon: # Get an random Action during traing
                action = env.action_space.sample() # Possible actions: {0: left, 1: down, 2: right, 3: up}
            else: # Lookup Q_Table
                action = np.argmax(q[state,:])
            # Take action, get updated state and rewards and info. Rweard is 1 if the agent has reached the goal, 0 otherwise
            new_state, reward, terminated, truncated, info = env.step(action)

            if is_training: # Calculate Q values for given state and action
                q[state, action] = q[state, action] + alpha_learning_rate * (
                    reward + gamma_discount_factor * np.max(q[new_state,:]) - q[state, action]
                )
            
            # Update state
            state = new_state

        # Decrease epsilon until it reaches 0. This ensures our model will explore and then apply what what it has learned.
        epsilon = max(epsilon-epsilon_decay_rate,0)
            
        # Decrease alpha rate after agent is done learning. This will help stabilize the Q values after the agent is done learning.
        if epsilon == 0:
            alpha_learning_rate = 0.0001

        # Store rewards for each episode
        if reward == 1:
            rewards_per_episode[i] = 1

    # Close the environment
    env.close()
    
    # Calculate total reward for every 100 episodes
    sum_rewards =np.zeros(episodes)
    for i in range(episodes):
        sum_rewards[i] = np.sum(rewards_per_episode[max(0, i-100):(i+1)]) # Calculate the sum of rewards for the last 100 episodes

    # Visualize the results
    plt.plot(sum_rewards) # Plot the rewards
    #plt.title("Rewards per 100 episodes: Q-Learning on Non slip FrozenLake - ADD EPSILON DECAY")
    #plt.title("Rewards per 100 episodes: Q-Learning on Slippery FrozenLake")
    plt.title("Rewards per 100 episodes:  Q-Table on Slippery FrozenLake")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per 100 epsisodes")
    #plt.savefig("FrozenLake_Q_nonslip.png") # Save the plot
    plt.savefig(os.path.join("QTable", "FrozenLake_QTable_slip.png")) # Save the plot


    if is_training:# Only save the Q-table while training
        #f = open("FrozenLake_Qtable_nonslip.pkl", "wb")
        f = open(os.path.join("QTable", "FrozenLake_Qtable.pkl"), "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == "__main__":
    run(10000, is_training=True,render=False) # Run the training for 15,000 episodes without rendering
    
    run(2, is_training=False,render=True) # Run the trained agent for 2 episodes with rendering