import numpy as np

class PPORolloutBuffer:
    """
    This class is a buffer used to store the experiences (state, action, log probability, reward, done flag) during
    an episode for the PPO (Proximal Policy Optimization) algorithm. The buffer stores the transitions collected during
    an episode until the episode is finished. After that, it is used to update the policy using the PPO algorithm.

    The buffer has a maximum size (`max_size`), and once it is full, no new transitions can be stored until it is cleared.
    The data is stored in numpy arrays and is used to perform policy and value updates on the PPO agent after each
    episode or batch of episodes.

    The main functionalities of the buffer are:
    1. Storing transitions (state, action, log probability, reward, and terminal flag) during an episode.
    2. Clearing the buffer when starting a new set of episodes.
    3. Returning the stored transitions in a form suitable for PPO updates.
    """
    def __init__(self, max_size, input_dims, n_actions):
        """
        Initializes the PPO rollout buffer. The buffer is used to store states, actions, log probabilities, 
        rewards, and done flags for each time step during the agent's interaction with the environment.

        Arguments:
        max_size: The maximum number of transitions the buffer can store before it needs to be cleared.
        input_dims: The dimensions of the state space (observation space).
        n_actions: The number of actions the agent can take (size of the action space).
        """
        self.states = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.actions = np.zeros((max_size, n_actions), dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.is_terminals = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = max_size

    def store_transition(self, state, action, log_prob, reward, done):
        """
        This function stores a transition (state, action, log probability, reward, and terminal flag) in the buffer.
        
        Arguments:
        state: The state (observation) at the current time step.
        action: The action taken at the current time step.
        log_prob: The log probability of the action taken (used in the PPO objective).
        reward: The reward received after taking the action.
        done: A flag indicating if the episode has ended (True if the episode is over).
        """
        if self.ptr >= self.max_size:
            raise Exception("PPO Buffer is full")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.is_terminals[self.ptr] = done
        self.ptr += 1

    def clear(self):
        """
        This function clears the buffer by resetting the pointer to 0. This is typically done when starting a new set of episodes.
        """
        self.ptr = 0

    def get(self):
        """
        This function returns the stored transitions in the buffer up to the current pointer position.
        
        Returns:
        A tuple containing:
        - states: The stored states (up to the current pointer).
        - actions: The stored actions (up to the current pointer).
        - log_probs: The stored log probabilities (up to the current pointer).
        - rewards: The stored rewards (up to the current pointer).
        - is_terminals: The stored done flags (up to the current pointer).
        """
        return (self.states[:self.ptr],
                self.actions[:self.ptr],
                self.log_probs[:self.ptr],
                self.rewards[:self.ptr],
                self.is_terminals[:self.ptr])