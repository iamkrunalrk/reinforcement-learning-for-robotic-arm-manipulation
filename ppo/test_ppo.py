import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from ppo_torch import PPOAgent
import torch

if __name__ == '__main__':

    """
    This is the main function where the entire training process for the PPO agent is set up and executed.

    The function:
    - Checks if CUDA (GPU) is available for training.
    - Creates necessary directories for saving models.
    - Sets up the environment using the `robosuite` library and wraps it using GymWrapper.
    - Initializes the PPO agent with specified hyperparameters.
    - Sets up TensorBoard for logging.
    - Loads pre-trained models (if available).
    - Runs the training loop for a specified number of episodes, where the agent interacts with the environment and learns from it.

    Main components:
    - CUDA availability check for selecting CPU or GPU for training.
    - Environment setup using `robosuite` with specific robot configuration (`Panda`) and settings like reward shaping and control frequency.
    - PPO agent initialization with hyperparameters for the reinforcement learning algorithm.
    - Training loop where the agent interacts with the environment, collects rewards, and updates its policy.
    """

    # Check if CUDA (GPU) is available
    print(torch.__version__)
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will use GPU.")
    else:
        print(f"CUDA is NOT available. Training will use CPU.")

    # The directory `tmp/ppo` is created if it doesn't exist. 
    # This directory will be used to store the PPO model's weights during training for saving and future loading.
    
    if not os.path.exists("tmp/ppo"):
        os.makedirs("tmp/ppo")
    else:
        pass  # No action needed if directory exists

    # Define environment
    """
    The environment for training is defined using `robosuite`. In this case, the "Door" environment is used,
    which involves a robot interacting with a door in a simulated environment. The `Panda` robot is selected to 
    perform the task, and various configurations for rendering, control frequency, and reward shaping are set.
    """
    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    # Hyperparameters
    """
    Various hyperparameters for training the PPO agent are set here:
    - `lr_actor` and `lr_critic`: Learning rates for the actor and critic networks.
    - `gamma`: Discount factor for future rewards.
    - `K_epochs`: Number of epochs to optimize the policy during each update.
    - `eps_clip`: Clipping parameter for the PPO objective function to control the magnitude of policy updates.
    - `buffer_size`: Size of the buffer that stores the collected experiences.
    - `batch_size`: The batch size used for training during each update.
    - `entropy_coeff`: Coefficient that weighs the entropy loss term to encourage exploration.
    """
    lr_actor = 3e-4
    lr_critic = 1e-3
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    buffer_size = 2048
    batch_size = 64
    entropy_coeff = 0.01

    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    # Initialize PPO agent
    """
    The PPO agent is initialized with the specified hyperparameters. The `PPOAgent` class takes the state 
    dimensions, action space, learning rates, and other parameters as arguments, and sets up the agent 
    to interact with the environment.
    """
    agent = PPOAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        buffer_size=buffer_size,
        batch_size=batch_size,
        entropy_coeff=entropy_coeff,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    # Initialize TensorBoard writer
    """
    TensorBoard is initialized here to log the training progress. It helps visualize metrics such as reward
    and loss during training. The logs are saved in the `logs/ppo` directory.
    """
    writer = SummaryWriter('logs/ppo')

    # Number of training episodes
    """
    The episode identifier is a string that specifies the configuration of the PPO agent during training,
    including learning rates and hyperparameters. It is used in logging and tracking purposes.
    """
    n_games = 3

    # Episode identifier for logging
    episode_identifier = f"PPO - lr_actor={lr_actor} lr_critic={lr_critic} K_epochs={K_epochs} eps_clip={eps_clip}"

    # Load existing models if available
    """
    Before starting the training loop, the agent attempts to load pre-trained models for the actor, critic, 
    and old policy (used for computing log probabilities). This allows the agent to resume training from a 
    checkpoint if available.
    """
    agent.load_models()

    # Start training loop
    """
    The training loop runs for a predefined number of episodes. In each episode:
    1. The environment is reset.
    2. The agent chooses an action based on the current state.
    3. The agent interacts with the environment, receiving the next state and reward.
    4. The score (total reward for the episode) is updated.
    
    After each episode, the total score is printed for monitoring progress.
    """

    for i in range(n_games):
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action, log_prob = agent.choose_action(state)
            next_state, reward, done, extra, info = env.step(action)
            env.render()
            score += reward

            state = next_state

        print(f"Episode: {i} Score: {score}")

