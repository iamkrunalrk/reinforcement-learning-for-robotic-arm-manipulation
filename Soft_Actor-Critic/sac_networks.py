import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    The ActorNetwork defines a neural network for the actor in the Soft Actor-Critic (SAC) algorithm. The actor network takes
    the state as input and outputs the mean and standard deviation of a Gaussian distribution. The mean determines the action's 
    central tendency, while the standard deviation controls the exploration/exploitation tradeoff by scaling the action space.
    The network consists of two fully connected layers followed by two output layers: one for the mean and one for the log standard deviation.

    The _init_weights method initializes the network's weights using the Kaiming normal initialization for the linear layers,
    and biases are initialized to zero.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the network layers using Kaiming normal initialization for the weights and zero initialization
        for the biases to help with efficient gradient flow.
        """
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.mean, self.log_std]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        """
        Forward pass through the actor network. The state is passed through two fully connected layers with ReLU activation.
        The output of the second layer is passed through two separate layers: one for the mean and one for the log standard deviation.
        The log standard deviation is clamped to prevent extremely large or small values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # To prevent too large or small std
        std = torch.exp(log_std)
        return mean, std


class CriticNetwork(nn.Module):
    """
    The CriticNetwork defines the neural network for the critic in the SAC algorithm. The critic is responsible for estimating the Q-values,
    which represent the expected return (reward) for a given state-action pair. The network uses two Q-value estimators (Q1 and Q2) to reduce 
    overestimation bias. Each Q-value estimator is composed of two fully connected layers, with one output for each Q-value estimator.
    
    The network uses two separate Q-value estimators (Q1 and Q2) to improve training stability. Each forward pass through the network computes 
    Q-values for both estimators, and the target is usually the minimum of these two values (Double Q-Learning).
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2, name='critic'):
        super(CriticNetwork, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # Q2 architecture
        self.fc1_2 = nn.Linear(*input_dims, fc1_dims)
        self.fc2_2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.q_2 = nn.Linear(fc2_dims, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the network layers using Kaiming normal initialization for the weights and zero initialization
        for the biases to help with efficient gradient flow.
        """
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.q, self.fc1_2, self.fc2_2, self.q_2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state, action):
        """
        Forward pass through the critic network. The state is passed through the first fully connected layer, then concatenated with the 
        action before passing through the second fully connected layer. This process is done for both Q1 and Q2 estimators.
        """
        # Q1 forward
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q1 = self.q(x)

        # Q2 forward
        x2 = F.relu(self.fc1_2(state))
        x2 = torch.cat([x2, action], dim=1)
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q_2(x2)

        return q1, q2

    def Q1(self, state, action):
        """
        A helper function to get only the Q1 value for a given state-action pair. This function is used when the Q1 value is needed
        independently of Q2 (e.g., for target computation or value propagation).
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q1 = self.q(x)
        return q1