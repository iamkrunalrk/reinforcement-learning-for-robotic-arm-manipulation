import os
import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

def make_env(seed=42):
    """
    Creates and wraps the Robosuite Door environment.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        VecNormalize: A normalized, vectorized environment compatible with SB3.
    """
    # Environment Configuration
    ENV_NAME = "Door"
    ROBOT = "Panda"
    CONTROLLER = "JOINT_VELOCITY"
    HAS_RENDERER = False
    USE_CAMERA_OBS = False
    HORIZON = 300
    REWARD_SHAPING = True
    CONTROL_FREQ = 20

    # Create the Robosuite environment
    env = suite.make(
        env_name=ENV_NAME,
        robots=ROBOT,
        controller_configs=suite.load_controller_config(default_controller=CONTROLLER),
        has_renderer=HAS_RENDERER,
        use_camera_obs=USE_CAMERA_OBS,
        horizon=HORIZON,
        reward_shaping=REWARD_SHAPING,
        control_freq=CONTROL_FREQ
    )

    # Wrap the environment to make it compatible with OpenAI Gym
    env = GymWrapper(env)

    # Vectorize the environment
    env = DummyVecEnv([lambda: env])

    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Print action space details for debugging
    print("\n===== Environment Action Space =====")
    print("Action Space:", env.action_space)
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Low:", env.action_space.low)
    print("Action Space High:", env.action_space.high)
    print("====================================\n")

    return env

def main():
    # ===========================
    # 1. Set Random Seeds
    # ===========================
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # ===========================
    # 2. Initialize Environments
    # ===========================
    env = make_env(seed=SEED)
    eval_env = make_env(seed=SEED + 1)  # Separate environment for evaluation

    # ===========================
    # 3. Define TD3 Hyperparameters
    # ===========================

    # Define action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Define custom network architecture
    policy_kwargs = dict(
        net_arch=[256, 128]  # Two hidden layers with 256 neurons each
    )

    TD3_HYPERPARAMS = {
        'policy': 'MlpPolicy',
        'policy_kwargs': policy_kwargs,
        'env': env,
        'learning_rate': 1e-3,          # Typical starting point
        'buffer_size': 1_000_000,       # Replay buffer size
        'learning_starts': 1_000,       # Number of steps before learning starts
        'batch_size': 256,               # Increased batch size for stability
        'gamma': 0.99,                   # Discount factor
        'tau': 0.005,                    # Soft update coefficient
        'policy_delay': 2,               # Delay policy updates
        'target_policy_noise': 0.1,             # Noise added to target actions
        'target_noise_clip': 0.5,        # Clipping range for target noise
        'action_noise': action_noise,  # Added action noise
        'train_freq': (1, 'step'),       # Frequency of training
        'gradient_steps': 1,             # Gradient steps per training
        'verbose': 1,
        'tensorboard_log': "./logs/td3_training",
        'seed': SEED,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    }

    # ===========================
    # 4. Initialize the TD3 Agent
    # ===========================
    model = TD3(**TD3_HYPERPARAMS)

    # Verify the agent's action space
    print("\n===== Agent Action Space =====")
    print("Agent Action Space:", model.action_space)
    print("Agent Action Space Shape:", model.action_space.shape)
    print("Agent Action Space Low:", model.action_space.low)
    print("Agent Action Space High:", model.action_space.high)
    print("====================================\n")

    # ===========================
    # 5. Set Up Logging and Callbacks
    # ===========================
    log_dir = TD3_HYPERPARAMS['tensorboard_log']
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer (optional, SB3 handles logging via tensorboard_log)
    writer = SummaryWriter(log_dir=log_dir)

    # Callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval_logs'),
        eval_freq=10_000,                # Evaluate every 10,000 steps
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Callback for checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,                 # Save every 50,000 steps
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='td3_checkpoint'
    )

    # ===========================
    # 6. Train the TD3 Agent
    # ===========================
    total_timesteps = 500_000  # Increased to allow sufficient training

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10  # Log every 10 episodes
    )

    # ===========================
    # 7. Save the Final Model
    # ===========================
    model.save("models/td3_final_model")
    print("Training completed and model saved.")

    # ===========================
    # 8. Cleanup
    # ===========================
    env.close()
    eval_env.close()
    writer.close()

if __name__ == "__main__":
    main()