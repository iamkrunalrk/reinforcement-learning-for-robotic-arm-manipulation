import os
import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

def make_env(seed=42):
    # Environment Configuration
    ENV_NAME = "Door"
    ROBOT = "Panda"  # SB3 expects a single robot, not a list
    CONTROLLER = "JOINT_VELOCITY"
    HAS_RENDERER = False
    USE_CAMERA_OBS = False
    HORIZON = 300
    REWARD_SHAPING = True
    CONTROL_FREQ = 20

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

    env = GymWrapper(env)
    env = DummyVecEnv([lambda: env])  # SB3 requires vectorized environments
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

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
    # 3. Define SAC Hyperparameters
    # ===========================
    SAC_HYPERPARAMS = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 3e-4,          # Corresponds to LR_ACTOR and LR_CRITIC
        'buffer_size': 1_000_000,       # MAX_SIZE
        'learning_starts': 1_000,       # INITIAL_RANDOM_STEPS
        'batch_size': 256,               # BATCH_SIZE (increased for stability)
        'gamma': 0.98,                   # GAMMA
        'tau': 0.005,                    # TAU
        'ent_coef': 'auto',              # Automatically tune entropy coefficient (corresponds to LR_ALPHA)
        'target_update_interval': 1,     # Update target network every step
        'train_freq': (1, 'step'),       # Train every step
        'gradient_steps': 1,             # One gradient step per environment step
        'verbose': 1,
        'tensorboard_log': "./logs/sac_training",
        'seed': SEED,
        'target_entropy': 'auto',        # Let SB3 set target entropy automatically
    }

    # ===========================
    # 4. Initialize the SAC Agent
    # ===========================
    model = SAC(**SAC_HYPERPARAMS)

    # ===========================
    # 5. Set Up Logging and Callbacks
    # ===========================
    log_dir = SAC_HYPERPARAMS['tensorboard_log']
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer (optional, SB3 handles logging via tensorboard_log)
    writer = SummaryWriter(log_dir=log_dir)

    # Callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval_logs'),
        eval_freq=10_000,                # Evaluate every 10,000 steps
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,                 # Save every 50,000 steps
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='sac_checkpoint'
    )

    # ===========================
    # 6. Train the SAC Agent
    # ===========================
    total_timesteps = 500_000  # Adjust based on your computational resources

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10  # Log every 10 episodes
    )

    # ===========================
    # 7. Save the Final Model
    # ===========================
    model.save("models/sac_final_model")
    print("Training completed and model saved.")

    # Close environments and writer
    env.close()
    eval_env.close()
    writer.close()

if __name__ == "__main__":
    main()