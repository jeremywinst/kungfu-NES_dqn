# """
# Define discrete action spaces for Gym Retro environments with a limited set of button combos
# """
#
# import gym
# import numpy as np
# import retro
#
# class Discretizer(gym.ActionWrapper):
#     """
#     Wrap a gym environment and make it use discrete actions.
#     Args:
#         combos: ordered list of lists of valid button combinations
#     """
#
#     def __init__(self, env, combos):
#         super().__init__(env)
#         assert isinstance(env.action_space, gym.spaces.MultiBinary)
#         buttons = env.unwrapped.buttons
#         self._decode_discrete_action = []
#         for combo in combos:
#             arr = np.array([False] * env.action_space.n)
#             for button in combo:
#                 arr[buttons.index(button)] = True
#             self._decode_discrete_action.append(arr)
#
#         self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))
#
#     def action(self, act):
#         return self._decode_discrete_action[act].copy()
#
#
# class KungFuDiscretizer(Discretizer):
#     """
#     A = Punch (higher reward)
#     B = Kick
#     """
#     def __init__(self, env):
#         super().__init__(env=env, combos=[['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['B'], ['DOWN', 'B']])
#
#
# def main():
#     env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.DISCRETE)
#     print('retro.Actions.DISCRETE action_space', env.action_space)
#     env.close()
#
#     env = retro.make(game='KungFu-Nes')
#     env = KungFuDiscretizer(env)
#     print('SonicDiscretizer action_space', env.action_space)
#     env.close()
#
#
# if __name__ == '__main__':
#     main()

# act = [0,0,0,0,1]
#
# for n, i in enumerate(act):
#     if i == 1 :
#         print(n)

import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# x = np.array([5, 4, 1, 4, 5])
# y = np.sort(x)
#
# plt.title("Line graph")
# plt.plot(x, y, color="red")
#
# plt.show()


import numpy
a = numpy.array([1, 2, 3])
t = torch.from_numpy(a)