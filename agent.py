# import flappy_bird_gymnasium
import torch
import torch.nn as nn
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import itertools 
import random
import os
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import argparse
import numpy as np
import flappy_bird_gymnasium

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
class Agent:

    def __init__(self,hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        with open("hyperparameters.yaml", "r") as f:
            all_hyperparameters = yaml.safe_load(f)
        hyperparameters = all_hyperparameters[hyperparameter_set]
        self.replay_memory_size  = hyperparameters['replay_memory_size']  # size of replay memory
        self.mini_batch_size     = hyperparameters['mini_batch_size']     # size of the training data set sampled from the replay memory
        self.epsilon_init        = hyperparameters['epsilon_init']        # 1 = 100% random actions
        self.epsilon_decay       = hyperparameters['epsilon_decay']       # epsilon decay rate
        self.epsilon_min         = hyperparameters['epsilon_min']         # minimum epsilon value
        self.network_sync_rate   = hyperparameters['network_sync_rate']   # number of steps before syncing policy => target network
        self.learning_rate_a       = hyperparameters['learning_rate_a']
        self.discount_factor_g      = hyperparameters['discount_factor_g']
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_id = hyperparameters['env_id']  
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True,render=False):
        print("Using environment ID:", self.env_id)
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        # env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        epsilon_history = []
        rewards_per_episode = []
        if is_training:
            memory = ReplayMemory(10000)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
        
            # tracks the number of steps taken. Used for syncing policy => target network
            step_count = 0

            # Intialize optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.0001)

            best_reward = -999999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
        for episodes in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)  
            episode_reward = 0.0
            terminated = False
            while(not terminated and episode_reward < self.stop_on_reward):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        
                # Next action:
                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float).to(device)
                reward = torch.tensor(reward, dtype=torch.float).to(device)  # ✅ Fixes typo 'tesnor' and 'dytpe'
                # Accumlate reward
                episode_reward+=reward
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1
                # (feed the new state to your agent here)
                state = new_state

            rewards_per_episode.append(episode_reward)
            if is_training:
                epsilon_history.append(epsilon)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episodes}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


            # If enough experience has been collected
            if len(memory) > self.mini_batch_size and is_training:

                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate and is_training:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
                        
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack and stack the batch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Convert to batched tensors and move to device
        states       = torch.stack(states).to(device)
        actions      = torch.stack(actions).to(device)
        new_states   = torch.stack(new_states).to(device)
        rewards      = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(device)

        gamma = self.discount_factor_g  # discount factor

        with torch.no_grad():
            # Compute max Q-values from target DQN for the next states
            max_next_q = target_dqn(new_states).max(dim=1, keepdim=True)[0]

            # Compute target Q-values using the Bellman equation
            target_q = rewards.unsqueeze(1) + (1 - terminations.unsqueeze(1)) * gamma * max_next_q

        # Get predicted Q-values from the policy network for the taken actions
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1))

        # Compute loss between predicted and target Q-values
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)