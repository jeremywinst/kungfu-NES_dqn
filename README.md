## Playing Atari Game with DQN Agent
## Introduction
Welcome to my Reinforcement Learning project for playing Atari games! This repository contains code and resources for training agents to play Kungfu-NES games using reinforcement learning techniques. This project is based on the OpenAI Gym Retro with a focus on the Implementation of the Deep Q-Network (DQN) algorithm. It also supports a variety of Atari that are available in the Gym Retro platform.

<p align="center">
  <img src="/figures/before_training.gif" style="width:350px;"/>
  <img src="/figures/after_training.gif" style="width:350px;"/>
</p>

## Data Preprocessing
To address computational and memory requirements a series of preprocessing steps applied to raw images from Gym Retro frames, specifically images with dimensions of 224x240 pixels and a 128-color palette. 

<p align="center">
  <img src="/figures/data_preprocessing.jpg" style="width:350px;"/>
</p>

Here's a breakdown of the mentioned preprocessing steps:
- Conversion to Grayscale: Reduces the dimensionality of the data and simplifies subsequent processing.
- Cropping: The upper part of the screen, which includes elements like score, health, lives, and time, is removed. This is done because this information is deemed irrelevant to the task at hand.
- Pixel Value Normalization: The pixel values of the image are normalized, often to a range between 0 and 1. Normalization ensures that all pixel values are on a consistent scale.
- Resizing: The frame is resized from its original dimensions to a smaller size of 84x84 pixels. This further reduces the computational and memory requirements.
- Stacking Frames: To provide the agent with a sense of motion and temporal information, four frames are stacked together. This step is crucial for the agent to perceive changes over time.

## Model Architecture
<p align="center">
  <img src="/figures/DQN_arch.png" style="width:700px;"/>
</p>

The model processes an 84x84x4 image through three convolutional layers with ReLU activation functions. The layers consist of 32 filters (8x8, stride 4), 64 filters (4x4, stride 2), and 64 filters (3x3, stride 1). The flattened features are passed to a fully connected layer with 512 hidden units and an output of 9 units, corresponding to possible game actions.

Hyperparameters include using the RMSProp algorithm, a minibatch size of 32, a learning rate of 0.00025, and a momentum value of 0.95. The behavior policy is ε-greedy with ε linearly decreasing from 1 to 0.1 over training episodes. Replay memory size is 1 million, and experiences are sampled every 50 frames. Target policy updates use soft updates with τ = 0.0001, and a discount factor of 0.99 is employed.

Regarding rewards, killing purple, blue, and boss enemies yields +100, +500, and +2000, respectively. A custom punishment of -50 is applied if the player's health decreases. Training spans 2000 episodes. The behavior policy follows an ε-greedy approach, promoting exploration for the first 100 episodes and linearly decreasing ε to 0.1 thereafter.
