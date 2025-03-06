import numpy as np


class ReplayBuffer():
    """
    This class implements a Replay Buffer, which is an essential component of reinforcement learning algorithms like Deep Q-Learning or TD3. 
    The Replay Buffer stores experiences (state, action, reward, next state, done) encountered by the agent during its interactions with the environment.
    By storing these experiences, the agent can sample random batches of experiences for training, which helps break the correlation between consecutive experiences and leads to more stable learning.
    """

    def __init__(self, max_size, input_shape, n_actions):
        """
        Initializes the replay buffer with a given maximum size, input shape, and number of actions.
        The replay buffer stores the states, actions, rewards, next states, and done flags.
        It also keeps track of how many experiences have been stored using `mem_counter`.

        Parameters:
        max_size (int): The maximum number of experiences that the buffer can store.
        input_shape (tuple): The shape of the state space (the dimensions of the state).
        n_actions (int): The number of possible actions the agent can take.
        """
        self.mem_size = max_size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        """
        This function stores a single experience (state, action, reward, next_state, done) in the buffer.
        If the buffer is full, it overwrites the oldest experience based on the `mem_counter`.

        Parameters:
        state (ndarray): The current state the agent is in.
        action (ndarray): The action taken by the agent in the given state.
        reward (float): The reward received from the environment after taking the action.
        next_state (ndarray): The state the agent transitions to after taking the action.
        done (bool): Flag indicating if the episode has ended (True if the episode ends, False otherwise).
        """
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        """
        This function samples a random batch of experiences from the replay buffer. It ensures that the batch size does not exceed the number of experiences stored (i.e., the current memory size). The experiences are returned as a tuple of (states, actions, rewards, next states, done flags).

        Parameters:
        batch_size (int): The number of experiences to sample from the buffer.

        Returns:
        tuple: A tuple containing the sampled states, actions, rewards, next states, and done flags.
        """
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
