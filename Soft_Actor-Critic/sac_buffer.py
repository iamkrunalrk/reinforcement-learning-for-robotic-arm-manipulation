import numpy as np
import torch


class ReplayBuffer:
    """
    The ReplayBuffer class stores experiences (state, action, reward, next_state, done) during the agent's interaction
    with the environment. This allows the agent to sample batches of experiences for training, a technique often used 
    in reinforcement learning to break the correlation between consecutive experiences and improve training stability.
    The buffer maintains a fixed size, and older experiences are overwritten when the buffer reaches its maximum size.

    Attributes:
    - max_size: Maximum number of experiences the buffer can store.
    - ptr: Pointer to the current index where the next experience will be stored.
    - size: The current number of experiences stored in the buffer.
    - state_memory: Array to store states.
    - new_state_memory: Array to store next states (i.e., the result of an action taken).
    - action_memory: Array to store actions taken by the agent.
    - reward_memory: Array to store rewards obtained from the environment after taking an action.
    - done_memory: Array to store the done flag (whether the episode has finished or not).
    """

    def __init__(self, max_size, input_dims, n_actions, device=torch.device('cpu')):
        """
        Initializes the ReplayBuffer with the specified parameters.
        The buffer's memory is pre-allocated using numpy arrays.

        Args:
        - max_size: Maximum number of transitions the buffer can store.
        - input_dims: Dimensions of the input state (environment observation space).
        - n_actions: Number of possible actions in the environment (action space).
        - device: The device (CPU or GPU) where the buffer will be stored.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        """
        This function stores a single experience (transition) in the replay buffer. If the buffer is full, it overwrites the oldest
        experience.

        Args:
        - state: The state the agent was in before taking the action.
        - action: The action the agent took.
        - reward: The reward the agent received after taking the action.
        - state_: The new state the agent transitions to after taking the action.
        - done: A boolean flag indicating if the episode has ended after taking the action.
        """
        index = self.ptr % self.max_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.done_memory[index] = done
        self.ptr += 1
        if self.size < self.max_size:
            self.size += 1

    def sample_buffer(self, batch_size):
        """
        Samples a batch of experiences from the buffer. These experiences are used for training the agent.

        Args:
        - batch_size: The number of experiences to sample.

        Returns:
        - states: A tensor containing the states from the sampled batch.
        - actions: A tensor containing the actions from the sampled batch.
        - rewards: A tensor containing the rewards from the sampled batch.
        - states_: A tensor containing the next states from the sampled batch.
        - dones: A tensor containing the done flags from the sampled batch.
        """
        max_mem = self.size
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch]).to(self.device)
        actions = torch.tensor(self.action_memory[batch]).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch]).unsqueeze(1).to(self.device)
        states_ = torch.tensor(self.new_state_memory[batch]).to(self.device)
        dones = torch.tensor(self.done_memory[batch]).unsqueeze(1).to(self.device)

        return states, actions, rewards, states_, dones

    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.
        """
        return self.size