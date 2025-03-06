import os
import torch
import torch.cuda
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    """
    The `CriticNetwork` is part of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
    The critic network is responsible for approximating the Q-function, which evaluates the quality of a given action
    taken in a particular state. It uses a deep neural network architecture to take in the state and action as inputs
    and output a Q-value, which is used to guide the actor network during training.

    The `CriticNetwork` receives the current state of the environment and the action taken by the agent.
    It then processes this information through two fully connected layers (`fc1` and `fc2`) and computes the Q-value (`q1`).
    """

    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128,
                 name='critic',
                 checkpoint_dir='tmp/td3',
                 learning_rate=0.001):
        """
        Initializes the Critic Network with given parameters, including input dimensions, number of actions,
        and the number of neurons in each layer. The network architecture includes two fully connected layers
        followed by an output layer that computes the Q-value.
        
        Parameters:
        input_dims (tuple): The dimensions of the state input.
        n_actions (int): The number of actions the agent can take.
        fc1_dims (int): Number of neurons in the first fully connected layer.
        fc2_dims (int): Number of neurons in the second fully connected layer.
        name (str): The name of the model (used for saving/loading checkpoints).
        checkpoint_dir (str): The directory where model checkpoints will be stored.
        learning_rate (float): The learning rate for the optimizer.
        """
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)  # take the input of the env and action that the robot take this step
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)  # output of the critic network, which is the q value

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print(f"Create Critic Network on Device: {self.device}")

        self.to(self.device)

    def forward(self, state, action):
        """
        The forward method computes the Q-value for a given state-action pair.
        The input state and action are passed through the fully connected layers,
        and the final Q-value is computed.

        Parameters:
        state (Tensor): The state input to the network.
        action (Tensor): The action input to the network.

        Returns:
        Tensor: The Q-value representing the quality of the action taken in the given state.
        """
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)

        return q1

    def save_checkpoint(self):
        """
        This function saves the model's state (parameters) to the checkpoint file.
        It is used to store the trained model so that it can be loaded later.

        The checkpoint is saved as a `.pth` file in the `checkpoint_dir`.
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        This function loads the model's state (parameters) from the checkpoint file.
        It is used to restore a previously trained model.

        The checkpoint file should already exist in the `checkpoint_dir`.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    The `ActorNetwork` is another essential component of the TD3 algorithm. 
    It is responsible for selecting actions based on the current state of the environment.
    The actor network outputs a deterministic action (via the tanh activation) for the given state input.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=0.001,
                 n_actions=2,
                 name='actor',
                 checkpoint_dir='tmp/td3'):
        """
        Initializes the Actor Network with given parameters, including input dimensions, number of actions,
        and the number of neurons in each layer. The network consists of two fully connected layers,
        followed by an output layer that computes the action values.
        
        Parameters:
        input_dims (tuple): The dimensions of the state input.
        fc1_dims (int): Number of neurons in the first fully connected layer.
        fc2_dims (int): Number of neurons in the second fully connected layer.
        learning_rate (float): The learning rate for the optimizer.
        n_actions (int): The number of actions the agent can take.
        name (str): The name of the model (used for saving/loading checkpoints).
        checkpoint_dir (str): The directory where model checkpoints will be stored.
        """
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print(f"Create Actor Network on Device: {self.device}")

        self.to(self.device)

    def forward(self, state):
        """
        The forward method computes the action based on the given state.
        The state is passed through the fully connected layers, and the action is calculated.
        
        Parameters:
        state (Tensor): The state input to the network.

        Returns:
        Tensor: The action to be taken by the agent, scaled by a tanh activation function.
        """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = torch.tanh(self.output(x))

        return x

    def save_checkpoint(self):
        """
        This function saves the model's state (parameters) to the checkpoint file.
        It is used to store the trained model so that it can be loaded later.

        The checkpoint is saved as a `.pth` file in the `checkpoint_dir`.
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        This function loads the model's state (parameters) from the checkpoint file.
        It is used to restore a previously trained model.

        The checkpoint file should already exist in the `checkpoint_dir`.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))

