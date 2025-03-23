# Reinforcement Learning for Robotic Arm Manipulation


This repository contains the code to train a robotic arm to open a door in the Robosuite environment using reinforcement learning (RL) techniques.

The goal of this project is to develop an RL-based agent capable of opening a door, which requires precise control and manipulation of the robotic arm. The project uses several RL algorithms to accomplish this task, such as TD3 (Twin Delayed Deep Deterministic Policy Gradient), PPO (Proximal Policy Optimization), and SAC (Soft Actor-Critic).

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Training the Agent](#training-the-agent)
    - [Testing the Agent](#testing-the-agent)
4. [Algorithms Implemented](#algorithms-implemented)
5. [Environment Setup](#environment-setup)
6. [About](#about)

---

## Overview

This project uses the **Robosuite** simulator to simulate a robotic manipulator performing the task of opening a door. Robosuite is a simulation framework built on top of MuJoCo that allows easy configuration of robotic environments. The project employs **reinforcement learning** (RL) techniques to train a robotic arm to learn how to perform this task autonomously.

The main focus of this repository is on the **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm, which has been successfully trained to achieve the goal. However, other algorithms like **PPO (Proximal Policy Optimization)** and **SAC (Soft Actor-Critic)** are also included in the code but have not been thoroughly tested for the door-opening task.

---

## Installation

To get started with the project, clone this repository and install the required dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/iamkrunalrk/Reinforcement-Learning-Robotic-Arm.git
cd Reinforcement-Learning-Robotic-Arm
```

### 2. Install Dependencies

Make sure you have **Python 3.6+** installed. It is highly recommended to use a **virtual environment** or **Conda environment** to manage the dependencies for this project.

To install the required dependencies, run:

```bash
pip3 install -r requirements.txt
```

### 3. PyTorch Installation

The project uses **PyTorch** for training the RL agent. You can install PyTorch according to your system's specifications. For installation instructions, visit the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---

## Usage

The main steps for using this project are to train the agent and test its performance in the Robosuite environment. 

### Training the Agent

To train the robotic arm to open the door, you can run the provided script. The TD3 algorithm is used for training by default.

1. Navigate to the `TD3` directory:

   ```bash
   cd TD3
   ```

2. Run the following command to start the training process:

   ```bash
   python3 main.py
   ```

   The training process will begin, and the agent will attempt to learn how to manipulate the robotic arm to open the door. During training, the agent will interact with the environment and learn from its actions via reinforcement learning.

3. Training will take a significant amount of time (depending on your hardware) and will involve a large number of iterations (episodes). After around 1000 steps, the agent should start to show significant improvement and be able to open the door.

4. The trained model will be saved periodically during the training process.

### Testing the Agent

Once the agent has been trained, you can test its performance by running the `test.py` script.

1. Navigate to the root directory (if not already there):

   ```bash
   cd TD3
   ```

2. Run the following command to test the trained model:

   ```bash
   python3 test.py
   ```

   This will load the trained model and execute it in the Robosuite environment, where the agent will attempt to open the door.

---

## Algorithms Implemented

This repository implements three popular reinforcement learning algorithms for training the agent. Only the TD3 implementation has been successfully tested to open the door. The others are included for experimentation purposes.

### 1. **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**

TD3 is a model-free, off-policy RL algorithm designed for continuous action spaces. It uses two Q-networks and a target policy smoothing mechanism to reduce the overestimation bias common in Q-learning algorithms.

The TD3 algorithm has been successfully tested for opening the door in the Robosuite environment.

### 2. **PPO (Proximal Policy Optimization)**

PPO is another on-policy reinforcement learning algorithm. It is known for its simplicity and effectiveness. However, the PPO implementation in this repository has not been tested as thoroughly as TD3 for the door-opening task.

### 3. **SAC (Soft Actor-Critic)**

SAC is an off-policy actor-critic algorithm that optimizes both the policy and value function. Like TD3, SAC is designed for continuous action spaces. The implementation here has not been thoroughly tested either.

---

## Environment Setup

To set up the Robosuite environment, you will need to follow the instructions from the [Robosuite Installation Guide](https://robosuite.ai/). It requires MuJoCo and other dependencies to be installed on your system.

Once Robosuite is set up, you should be able to run the environment directly from the code.

---


## About

This project was created to implement a robotic manipulator capable of opening a door in the Robosuite environment using reinforcement learning. The code was inspired by a tutorial by **Robert Cowher** and aims to showcase the use of RL algorithms in solving robotic manipulation tasks.
