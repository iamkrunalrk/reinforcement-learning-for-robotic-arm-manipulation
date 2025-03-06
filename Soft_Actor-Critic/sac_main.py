import os
import torch
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from sac_torch import SACAgent
import numpy as np
import random

def main():
    """
    This function contains the main training loop for the SAC (Soft Actor-Critic) agent using the Robosuite environment.
    The environment, model, and agent configurations are set up, followed by an initial exploration phase where the agent
    randomly explores the environment. After that, the agent enters the training loop where it learns to perform the task.
    The training process includes logging information to TensorBoard, saving models at regular intervals, and managing 
    the environment's reset and state transitions. The SAC agent improves its performance over time by interacting with the
    environment and updating its policies based on past experiences.
    """
    
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will use GPU.")
    else:
        print(f"CUDA is NOT available. Training will use CPU.")


    # Environment Configuration
    ENV_NAME = "Door"
    ROBOTS = ["Panda"]
    CONTROLLER = "JOINT_VELOCITY"
    HAS_RENDERER = False
    USE_CAMERA_OBS = False
    HORIZON = 300
    REWARD_SHAPING = True
    CONTROL_FREQ = 20

    # Training Configuration
    MAX_EPISODES = 10000
    MAX_TIMESTEPS = 300
    LOG_INTERVAL = 10  # Save models and log every 10 episodes

    # SAC Hyperparameters
    GAMMA = 0.98  # Discount factor
    TAU = 0.005  # Soft update parameter for target networks
    LR_ACTOR = 3e-4  # Learning rate for Actor network
    LR_CRITIC = 3e-4  # Learning rate for Critic networks
    LR_ALPHA = 0.001  # Learning rate for entropy alpha
    MAX_SIZE = 1000000  # Replay buffer size
    BATCH_SIZE = 128  # Mini-batch size for updates

    # Initial Exploration Configuration
    INITIAL_RANDOM_STEPS = 10000  # Number of initial random actions
    RANDOM_ACTION_PROB = 1.0  # Probability of taking a random action during exploration

    # =======================
    # Initialize Environment
    # =======================

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
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_alpha=LR_ALPHA,
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
    # Initial Exploration Phase
    # =======================

    print("Starting Initial Exploration Phase...")
    state, _ = env.reset()
    total_steps = 0
    initial_exploration_steps = INITIAL_RANDOM_STEPS

    while total_steps < initial_exploration_steps:
        # Sample random action
        action = env.action_space.sample()
        # Scale the random action
        scaled_action = action  # Assuming env.action_space.sample() already provides actions in the correct scale

        # Execute action in the environment
        next_state, reward, done, _, info = env.step(action)
        total_steps += 1

        # Store transition in Replay Buffer
        agent.store_transition(state, action, reward, next_state, done)

        # Update state
        state = next_state

        if done:
            state, _ = env.reset()

        # Logging every 1000 steps
        if total_steps % 1000 == 0:
            print(f"Initial Exploration: {total_steps}/{initial_exploration_steps} steps completed.")

    print("Initial Exploration Phase Completed.\nStarting Training Phase...")

    # =======================
    # Training Loop
    # =======================

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        score = 0

        for t in range(MAX_TIMESTEPS):
            # Decide whether to take a random action or follow the policy
            if total_steps < initial_exploration_steps:
                # During initial exploration steps, take random actions
                action = env.action_space.sample()
                scaled_action = action  # Assuming env.action_space.sample() already provides actions in the correct scale
            else:
                # After initial exploration, follow the policy
                action, log_prob = agent.choose_action(state)
                scaled_action = action  # Action is already scaled in choose_action

            # Execute action in the environment
            next_state, reward, done, _, info = env.step(scaled_action)
            score += reward
            total_steps += 1

            # Store transition in Replay Buffer
            agent.store_transition(state, scaled_action, reward, next_state, done)

            # Update state
            state = next_state

            if done:
                break

        # Update SAC agent after each episode
        agent.update()

        # Log the episode score and alpha to TensorBoard
        writer.add_scalar("SAC/Reward", score, global_step=episode)
        writer.add_scalar("SAC/Alpha", agent.alpha.item(), global_step=episode)


        # Save models and print progress at defined intervals
        if episode % LOG_INTERVAL == 0:
            agent.save_models()
            print(f"Episode: {episode} | Score: {score:.2f} | Alpha: {agent.alpha.item():.4f}")

    # Close the environment and TensorBoard writer
    env.close()
    writer.close()

if __name__ == '__main__':
    main()