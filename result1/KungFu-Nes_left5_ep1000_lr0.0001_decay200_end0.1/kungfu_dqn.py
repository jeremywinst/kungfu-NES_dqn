
import time
import gym
import retro
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import argparse
import os
from shutil import copyfile

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
import pandas as pd
import csv

# os.environ['CUDA_VISIBLE_DEVICES']='3'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--train', type=str2bool, default=True, help='Set to "True" for training, set to "False" for deploy the model')
parser.add_argument('--load_epoch', type=str, default='1000', help='Load which trained model')
parser.add_argument('--name', type=str, default='1000', help='Code name for the experiment')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='learning rate (default: 0.0001)')
parser.add_argument('--decay', type=int, default=100, metavar='N', help='Rate by which epsilon to be decayed (default: 100)')
parser.add_argument('--r_left', type=int, default=10, metavar='N', help='Reward for going left (default: 10)')
parser.add_argument('--episode', type=int, default=1000, metavar='N', help='Number of episode for training (default: 1000)')
parser.add_argument('--end', type=float, default=0.01, metavar='N', help='Lowest value of eps (default: 0.01)')
opt = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        for n, i in enumerate(act):
            if i == 1:
                a = n
                break
        return self._decode_discrete_action[a].copy()


class KungFuDiscretizer(Discretizer):
    """
    A = Punch (higher reward)
    B = Kick
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['B'], ['DOWN', 'B']])

GAME_NAME = 'KungFu-Nes'
TRIAL_NUMBER = opt.name
env = retro.make(game=GAME_NAME, scenario=None)
env = KungFuDiscretizer(env)
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
env.seed(0)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = opt.lr            # learning rate (0.0001)
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
REPLAY_AFTER = 10000   # After which threshold replay to be started
EPS_START = 1          # starting value of epsilon
EPS_END = opt.end      # Ending value of epsilon
EPS_DECAY = opt.decay  # Rate by which epsilon to be decayed (100)
TRAIN = opt.train      # option for training or testing
LOAD_EPOCH = opt.load_epoch
MOMENTUM = 0.90        # momentum of the RMS optimizer
FINAL_EXPLORATION = 100

dir = './result/{}_{}_ep{}_lr{}_decay{}_end{}'.format(GAME_NAME, TRIAL_NUMBER, opt.episode, LR, EPS_DECAY, EPS_END)

MODEL_DIR = [dir+'/policy_net', dir+'/target_net'] # directory to save and load policy model and target model respectively

print()
print(dir)
print('Learning Rate: ', LR)
print('Decay: ', EPS_DECAY)
print('Eps end: ', EPS_END)
print('Reward Left: ', opt.r_left)
print()

# Check whether the specified path exists or not
if not os.path.exists(dir) :
    os.makedirs(dir)

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY,
                 REPLAY_AFTER, DQNCnn, TRAIN, LOAD_EPOCH, MODEL_DIR, MOMENTUM)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
reward_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
e = [epsilon_by_epsiode(i) for i in range(1000)]

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.title("Line graph")
plt.plot(e, color="red")
plt.savefig( os.path.join(dir,'eps.jpg'))

# plt.show()

def stack_frames(frames, state, is_new = False):
    frame = preprocess_frame(state, (107, 176, 0, 239), 84)

    plt.imshow(frame, cmap='gray')
    # plt.show()

    frames = stack_frame(frames, frame, is_new)
    return frames

def train(n_episodes):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    env.viewer = None
    # env.render()
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0.0

        # if i_episode < FINAL_EXPLORATION :
        #     eps = EPS_START
        # else:
        #     eps = epsilon_by_epsiode(i_episode-FINAL_EXPLORATION)

        eps = epsilon_by_epsiode(i_episode)

        timestamp = 0

        while timestamp < 10000:
            # env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1

            if timestamp > 1:
                if(prev_state['health'] > info['health']):
                    #reward += (prev_state['health']-info['health'])*(-50)
                    reward += -50
                if action == 0:
                    reward += opt.r_left

            prev_state = info

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
        scores_window.append(score)  # save most recent score
        reward_window.append(reward)

        scores.append(score)  # save most recent score

        #print('\rEpisode {}/{}\t\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, n_episodes, np.mean(scores_window), eps), end="")
        print('Episode {}/{}\t\tScore: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, n_episodes, score, eps))

        # save model and score table every 100 episode
        if i_episode % 1 == 0:
            # save both model
            policy_path = MODEL_DIR[0]+'_{}.pth'.format(i_episode)
            target_path = MODEL_DIR[1]+'_{}.pth'.format(i_episode)
            torch.save(agent.policy_net.state_dict(), policy_path)
            torch.save(agent.target_net.state_dict(), target_path)

            df = pd.DataFrame(scores)

            fields = ['score']

            with open(dir + '/{}_{}_ep{}_lr{}_decay{}_end{}.csv'.format(GAME_NAME, TRIAL_NUMBER, opt.episode, LR, EPS_DECAY, EPS_END), 'w', newline='') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(df.values.tolist())

    return scores

def deploy(n_episodes):
    env.viewer = None
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0.0
        eps = 0.3
        timestamp = 0

        while timestamp < 10000:
            env.render()
            action = agent.deploy(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1
            if done:
                break
        scores_window.append(score)  # save most recent score

    scores.append(score)  # save most recent score

    print('Episode {}\tAverage Score: {:.2f}'.format(n_episodes, np.mean(scores_window)))

#if __name__ == '__main__' :
start = time.time()

if TRAIN:
    copyfile(os.path.basename(__file__), os.path.join('./result/{}_{}_ep{}_lr{}_decay{}_end{}'.format(GAME_NAME, TRIAL_NUMBER, opt.episode, LR, EPS_DECAY, EPS_END), os.path.basename(__file__)))
    print("Start training...")
    train(n_episodes=opt.episode)

else:
    print('Deploying agent...')
    deploy(n_episodes=3)

elapsed_time = time.time() - start
print("")
print('Training finish')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))