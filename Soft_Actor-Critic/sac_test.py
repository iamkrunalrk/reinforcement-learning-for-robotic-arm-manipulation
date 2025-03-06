import os
import torch
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from sac_torch import SACAgent
import numpy as np
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will use GPU.")
    else:
        print(f"CUDA is NOT available. Training will use CPU.")

    # Random Seed for Reproducibility
    SEED = 42

    # Environment Configuration
    ENV_NAME = "Door"
    ROBOTS = ["Panda"]
    CONTROLLER = "JOINT_VELOCITY"
    HAS_RENDERER = True
    USE_CAMERA_OBS = False
    HORIZON = 300
    REWARD_SHAPING = True
    CONTROL_FREQ = 20

    # Training Configuration
    MAX_EPISODES = 3
    MAX_TIMESTEPS = 300
    LOG_INTERVAL = 10  # Save models and log every 10 episodes

    # SAC Hyperparameters
    ALPHA = 0.4  # Initial Entropy coefficient (will be overridden if using auto-tuning)
    GAMMA = 0.98  # Discount factor
    TAU = 0.005  # Soft update parameter for target networks
    LR_ACTOR = 3e-4  # Learning rate for Actor network
    LR_CRITIC = 3e-4  # Learning rate for Critic networks
    MAX_SIZE = 1000000  # Replay buffer size
    BATCH_SIZE = 128  # Mini-batch size for updates

    # =======================
    # Initialize Environment
    # =======================

    # Set random seeds
    set_seed(SEED)

    # Create directories for saving models and logs
    os.makedirs("tmp/sac", exist_ok=True)
    os.makedirs("logs/sac", exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter('logs/sac')

    # Create Robosuite environment
    env = suite.make(
        ENV_NAME,
        robots=ROBOTS,
        controller_configs=suite.load_controller_config(default_controller=CONTROLLER),
        has_renderer=HAS_RENDERER,
        use_camera_obs=USE_CAMERA_OBS,
        horizon=HORIZON,
        reward_shaping=REWARD_SHAPING,
        control_freq=CONTROL_FREQ,
    )

    # Wrap the environment with GymWrapper for compatibility
    env = GymWrapper(env)

    # Extract environment dimensions
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # Calculate action scaling parameters
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    # =======================
    # Initialize SAC Agent
    # =======================

    # Define device (GPU if available, else CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize SAC agent with the defined hyperparameters and action scaling
    agent = SACAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        max_size=MAX_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        action_scale=action_scale,
        action_bias=action_bias,
        target_entropy=-n_actions  # For automatic entropy tuning
    )

    # Load existing models if available; otherwise, start from scratch
    agent.load_models()

    # =======================
    # Training Loop
    # =======================

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        score = 0

        for t in range(MAX_TIMESTEPS):
            # Choose action based on current state
            action, log_prob = agent.choose_action(state)

            # Execute action in the environment
            next_state, reward, done, _,  info = env.step(action)
            score += reward

            env.render()

            # Update state
            state = next_state

            if done:
                break

        # Update SAC agent after each episode
        agent.update()

        print(f"Episode: {episode} | Score: {score:.2f} | Alpha: {agent.alpha.item():.4f}")

    # Close the environment and TensorBoard writer
    env.close()
    writer.close()


if __name__ == '__main__':
    main()