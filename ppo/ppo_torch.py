import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks_ppo import ActorNetwork, CriticNetwork
from buffer_ppo import PPORolloutBuffer

class PPOAgent:

    """
    The PPOAgent class is a reinforcement learning agent that uses Proximal Policy Optimization (PPO) to learn and
    improve its policy. PPO is a popular on-policy reinforcement learning algorithm that updates the policy using
    clipped surrogate objective functions and a value function to estimate the advantage of actions taken.

    The agent consists of three main components:
    1. Actor Network: Generates action probabilities.
    2. Critic Network: Estimates the value of a state-action pair (state-value).
    3. Rollout Buffer: Stores experiences (state, action, reward, etc.) for the PPO algorithm.

    The agent is trained through interaction with an environment where it observes the state, selects actions, receives
    rewards, and updates its policy and value function accordingly.

    Attributes:
    - gamma: Discount factor for future rewards.
    - eps_clip: Clipping parameter for the PPO objective function to control policy updates.
    - K_epochs: Number of epochs for updating the policy during each training step.
    - batch_size: Batch size for training.
    - entropy_coeff: Coefficient for entropy in the loss function to encourage exploration.
    - device: The device (CPU or GPU) on which the model will be loaded and run.
    - actor: The actor network that generates the policy.
    - critic: The critic network that estimates the value function.
    - policy_old: A copy of the actor's previous policy used to compute log probabilities for the importance sampling.
    - optimizer_actor: Optimizer for the actor network.
    - optimizer_critic: Optimizer for the critic network.
    - buffer: The PPO rollout buffer that stores the experiences.
    - MseLoss: The mean squared error loss used to train the critic network.
    """


    def __init__(self,
                 input_dims,
                 n_actions,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 K_epochs=80,
                 eps_clip=0.2,
                 buffer_size=2048,
                 batch_size=64,
                 entropy_coeff=0.01,
                 device=None):
        
        """
        Initializes the PPOAgent by setting up the actor and critic networks, optimizers, and the rollout buffer.

        Arguments:
        input_dims: The dimensions of the input state space.
        n_actions: The number of possible actions the agent can take.
        lr_actor: Learning rate for the actor network.
        lr_critic: Learning rate for the critic network.
        gamma: Discount factor for future rewards.
        K_epochs: Number of epochs to update the policy in each training iteration.
        eps_clip: The clipping parameter used in the PPO objective function to ensure stable updates.
        buffer_size: The size of the PPO rollout buffer.
        batch_size: The batch size for training the networks.
        entropy_coeff: Coefficient to weight the entropy term in the objective function to encourage exploration.
        device: The device (CPU or GPU) for running the model.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff

        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Print device information
        print(f"Using device: {self.device}")

        # Initialize actor and critic networks
        self.actor = ActorNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)
        self.critic = CriticNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)

        # Print device for the models
        print(f"Actor model is on device: {next(self.actor.parameters()).device}")
        print(f"Critic model is on device: {next(self.critic.parameters()).device}")

        # Initialize old actor for calculating log probabilities
        self.policy_old = ActorNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize rollout buffer
        self.buffer = PPORolloutBuffer(buffer_size, input_dims, n_actions)

        # Loss function
        self.MseLoss = nn.MSELoss()

    def choose_action(self, state):

        """
        Given a state, the agent selects an action based on the current policy (actor network). 

        This function uses the old policy for exploration and chooses actions based on the probability distribution
        output by the actor network. It also applies the `tanh` transformation to the action to ensure the action 
        is bounded.

        Arguments:
        state: The current state of the environment.

        Returns:
        action_clipped: The action chosen by the agent, clipped within valid bounds.
        log_prob: The log probability of the action under the old policy.
        """

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_mean = self.policy_old(state)
        # For exploration, sample from the policy's distribution
        std = torch.ones_like(action_mean) * 0.5  # Fixed standard deviation; can be parameterized
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_clipped = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Adjust log_prob for tanh transformation
        log_prob -= (2*(torch.log(torch.tensor(2.0)) - action - nn.functional.softplus(-2*action))).sum(dim=-1)
        return action_clipped.cpu().numpy(), log_prob.cpu().numpy()

    def remember(self, state, action, log_prob, reward, done):
        """
        This function store the current transition (state, action, log_prob, reward, done) in the PPO rollout buffer.

        Arguments:
        state: The current state of the environment.
        action: The action taken by the agent.
        log_prob: The log probability of the action under the current policy.
        reward: The reward received from the environment after taking the action.
        done: A boolean indicating if the episode has ended.
        """
        self.buffer.store_transition(state, action, log_prob, reward, done)

    def compute_returns_and_advantages(self, rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95):
        """
        This function compute the returns and advantages for the current batch of experiences. 

        The returns are the discounted sum of future rewards, while the advantages are computed using Generalized
        Advantage Estimation (GAE) to reduce variance in the gradient estimates.

        Arguments:
        rewards: List of rewards from the environment.
        dones: List of booleans indicating whether an episode ended.
        values: List of state values estimated by the critic network for each step.
        next_values: List of the next state values.
        gamma: Discount factor for future rewards.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation.

        Returns:
        returns: The computed returns for each step.
        advantages: The computed advantages for each step.
        """
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        advantages = np.array(advantages)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        """
        This function updates the actor and critic networks by performing several optimization steps.

        This function retrieves the data stored in the PPO rollout buffer, computes the returns and advantages, 
        and updates both the actor and critic networks using the PPO algorithm. It does so for `K_epochs` iterations 
        for each batch.

        It computes the PPO objective function and optimizes the policy and value function using the gradients.
        """
        # Retrieve data from buffer
        states, actions, log_probs_old, rewards, dones = self.buffer.get()
        self.buffer.clear()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute state values
        with torch.no_grad():
            values = self.critic(states, actions).squeeze().cpu().numpy()
            # Estimate next state values (assuming next state is the last state)
            # Since it's on-policy, we don't have next actions from the current policy
            next_state = states[-1].unsqueeze(0)
            next_action = actions[-1].unsqueeze(0)
            next_value = self.critic(next_state, next_action).squeeze().cpu().numpy()
            next_values = np.append(values[1:], next_value)

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards.cpu().numpy(), dones.cpu().numpy(), values, next_values, self.gamma, 0.95)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Re-compute log probabilities and state values
            action_mean = self.actor(states)
            std = torch.ones_like(action_mean) * 0.5
            dist = Normal(action_mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Compute ratios for clipping
            ratios = torch.exp(log_probs - log_probs_old)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

            # Compute critic loss
            state_values = self.critic(states, actions).squeeze()
            critic_loss = self.MseLoss(state_values, returns)

            # Take gradient step for actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Take gradient step for critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # Update old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

    def save_models(self):
        torch.save(self.actor.state_dict(), 'tmp/ppo/actor.pth')
        torch.save(self.critic.state_dict(), 'tmp/ppo/critic.pth')
        torch.save(self.policy_old.state_dict(), 'tmp/ppo/policy_old.pth')
        print("PPO models saved successfully.")

    def load_models(self):
        try:
            self.actor.load_state_dict(torch.load('tmp/ppo/actor.pth'))
            self.critic.load_state_dict(torch.load('tmp/ppo/critic.pth'))
            self.policy_old.load_state_dict(torch.load('tmp/ppo/policy_old.pth'))
            print("PPO models loaded successfully.")
        except:
            print("Failed to load PPO models. Starting from scratch.")