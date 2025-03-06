import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    The ActorNetwork class defines a neural network used for the actor component in a reinforcement learning setup,
    particularly for algorithms like PPO (Proximal Policy Optimization). The role of the actor network is to produce
    a policy, which is typically represented by a probability distribution over actions given a state.

    This network takes a state as input and outputs the parameters of a probability distribution over actions (in this
    case, the mean action). The output of the network is passed through a `tanh` activation to ensure that the action
    stays within the valid range.

    The network consists of two fully connected layers followed by the output layer, which outputs the mean action.
    The actor network's goal is to optimize the policy based on the rewards it receives, allowing the agent to select
    actions that maximize its expected return.

    Attributes:
    - input_dims: The dimensions of the input state space.
    - fc1_dims: The number of neurons in the first hidden layer.
    - fc2_dims: The number of neurons in the second hidden layer.
    - n_actions: The number of actions the agent can take.
    - device: The device (CPU or GPU) where the model will be loaded and run.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128,
                 n_actions=2, name='actor', checkpoint_dir='tmp/ppo'):
        """
        Initializes the ActorNetwork with the given architecture and layer sizes. The network consists of two
        fully connected layers, and it outputs the mean action value after passing through a `tanh` activation.

        Arguments:
        input_dims: The dimensions of the input state space.
        fc1_dims: The number of neurons in the first fully connected layer.
        fc2_dims: The number of neurons in the second fully connected layer.
        n_actions: The number of actions the agent can take (size of the output layer).
        name: The name of the network.
        checkpoint_dir: The directory where the model checkpoints will be saved (not used in this code snippet).
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        self._init_weights()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """
        This function initializes the weights of the layers using the Kaiming normal initialization method, which is commonly used
        for layers with ReLU activations. The biases are initialized to zero.

        This helps the network learn efficiently by ensuring that the initial weights do not cause large gradient
        problems during training.
        """
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        """
        This function defines the forward pass of the actor network. The state is passed through the network layers, and the 
        output is a mean action value. The output is passed through a `tanh` activation to bound the actions within
        a valid range, typically between -1 and 1 for continuous action spaces.

        Arguments:
        state: The current state input for the network.

        Returns:
        action_mean: The mean action value output by the network, passed through a `tanh` function.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = torch.tanh(self.output(x))
        return action_mean


class CriticNetwork(nn.Module):
    """
    The CriticNetwork class defines a neural network used for the critic component in reinforcement learning. 
    The critic estimates the value of the current state and the action taken (state-action value function Q(s,a)).

    This network takes both the state and the action as inputs and outputs the estimated value of the state-action pair.
    The goal of the critic is to provide feedback to the actor on how good or bad the taken actions are, which is 
    used to optimize the policy.

    Attributes:
    - input_dims: The dimensions of the input state space.
    - fc1_dims: The number of neurons in the first hidden layer.
    - fc2_dims: The number of neurons in the second hidden layer.
    - n_actions: The number of actions the agent can take.
    - device: The device (CPU or GPU) where the model will be loaded and run.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128,
                 n_actions=2, name='critic', checkpoint_dir='tmp/ppo'):
        """
        Initializes the CriticNetwork with the given architecture and layer sizes. The network consists of two
        fully connected layers and outputs a scalar value that estimates the state-action value.

        Arguments:
        input_dims: The dimensions of the input state space.
        fc1_dims: The number of neurons in the first fully connected layer.
        fc2_dims: The number of neurons in the second fully connected layer.
        n_actions: The number of actions the agent can take (used to form the input for the second layer).
        name: The name of the network.
        checkpoint_dir: The directory where the model checkpoints will be saved (not used in this code snippet).
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.output = nn.Linear(fc2_dims, 1)

        # Initialize weights
        self._init_weights()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """
        This function initializes the weights of the layers using the Kaiming normal initialization method, which is commonly used
        for layers with ReLU activations. The biases are initialized to zero.

        This helps the network learn efficiently by ensuring that the initial weights do not cause large gradient
        problems during training.
        """
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state, action):
        """
        This function defines the forward pass of the critic network. The state and action are passed through the network to
        compute the value of the state-action pair. The value is used to evaluate how good or bad the taken action is.

        Arguments:
        state: The current state input for the network.
        action: The action taken by the agent at the given state.

        Returns:
        state_value: The estimated value of the state-action pair.
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        state_value = self.output(x)
        return state_value