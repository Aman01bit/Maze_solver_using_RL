
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])

class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer                                             # this is actually a reference (kindof replay buffer stores the agent past experiences and outcomes)
        self.num_act = 4                                                        # Action that will be taken Up, down, left , right.
        self.use_softmax = use_softmax                                          # how the agent selects the action -> e- greedy will be selected if false and otherwise softmax
        self.total_reward = 0                                                   # cumulative reward.
        self.min_reward = -self.env.maze.size                                   # used as lower bound so that if agent move endlessely and cumulative rewards gets negative.
        self.isgameon = True                                                    # tells if the game is still running.

    def make_a_move(self, net, epsilon, device = 'cuda'):
        action = self.select_action(net, epsilon, device)                       # epsilon controls randomness of picking the action i.e (will explore or exploit). neural network -> choose action decision based on strategy(e- greedy or softmax).
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)       # update the reward for the next state and game is on or not.
        self.total_reward += reward

        if self.total_reward < self.min_reward:                                 # if cumulative reward falls below my minimum reward (then i will make my game off).
            self.isgameon = False
        if not self.isgameon:                                                   # and will make the reward to zero.
            self.total_reward = 0

        transition = Transition(current_state, action,                          # We will make the transition.
                                next_state, reward,
                                self.isgameon)

        self.buffer.push(transition)                                            # Saving the experience of transition and we will use to train the DQN from these past experiences.

    def select_action(self, net, epsilon, device = 'cuda'):
        state = torch.Tensor(self.env.state()).to(device).view(1,-1)
        qvalues = net(state).cpu().detach().numpy().squeeze()                   # qvalues for each possible action.

        # softmax sampling of the qvalues
        if self.use_softmax:
            p = sp.softmax(qvalues/epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p = p)

        # else choose the best action with probability 1-epsilon
        # and with probability epsilon choose at random
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:
                action = np.argmax(qvalues, axis=0)
                action = int(action)

        return action


    def plot_policy_map(self, net, filename, offset):                                 # Plotting the maze.
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')

            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
                qvalues = net(torch.Tensor(self.env.state()).view(1,-1).to('cuda'))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                policy = self.env.directions[action]

                ax.text(free_cell[1]-offset[0], free_cell[0]-offset[1], policy)
            ax = plt.gca();

            plt.xticks([], [])
            plt.yticks([], [])

            ax.plot(self.env.goal[1], self.env.goal[0],
                    'bs', markersize = 4)
            plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
            plt.show()