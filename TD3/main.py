import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import torch

if __name__ == '__main__':
    """
    The main entry point for running the TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent in a robosuite environment. 
    This script sets up the environment, initializes the agent, and runs the agent through a number of episodes to test its performance. 
    The environment is based on the "Door" task in robosuite, where the agent's goal is to manipulate a robotic arm (Panda) to perform a task
    (opening a door).
    """
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will use GPU.")
    else:
        print(f"CUDA is NOT available. Training will use CPU.")

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")
    else:
        pass

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size)
    #print(env.observation_space.shape)

    writer = SummaryWriter('logs')
    n_games = 10000
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_leanring_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        # print(f"Observation  after reset: {observation}")

        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, extra, info = env.step(action)
            # print(f"ob:{next_observation},\n reward: {reward},\n done {done}, \ninfo:{info}, \n extra: {extra}")
            score += reward
            # print(f"state space shape: {observation.shape}, state_memory shape: {agent.memory.state_memory[0].shape}")
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()

            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if i % 10 == 0:
            agent.save_models()

        print(f"Episode: {i} Score: {score}")