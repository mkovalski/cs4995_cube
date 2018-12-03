#!/usr/bin/env python

import numpy as np
from rubiks import Cube
from network import ADINetwork
import tensorflow as tf
import copy
import argparse
import pdb
import sys

class Node(object):
    ''' A node in the MCTS algorithm. Contains information such as number
    of visits, etc.
    
    '''
    def __init__(self, parent, cube, exploration, virt_loss, network):
        
        num_actions = 12

        self.parent = parent
        self.cube = cube
        self.c = exploration
        self.v = virt_loss
        
        _, self.p_s = network.evaluate(self.cube.cube.reshape(1, -1))

        # Number of times an action a has been taken from state
        self.n_s = np.zeros(num_actions)

        # Maximal value of action a from state s
        self.w_s = np.zeros(num_actions)
        
        # The current virtual loss for action a from state s 
        self.l_s = np.zeros(num_actions)

        self.children = np.asarray([None] * num_actions)
        
    def UCT(self):
        return (self.c * self.p_s * np.sqrt(np.sum(self.n_s)))/(1 + self.n_s)
    
    def QCT(self):
        q = self.w_s - self.l_s
        self.l_s += self.v
        return q

    def policy(self):
        return np.argmax(self.UCT() + self.QCT())

    def is_leaf(self):
        return np.all(self.children == None)

    def expand(self, network):
        actions = np.eye(12)
        for i in range(actions.shape[0]):
            tmp_cube = copy.deepcopy(self.cube)
            tmp_cube.move(actions[i, :])
            self.children[i] = Node(self, tmp_cube, self.c, self.v, network)
        
        value, _= network.evaluate(self.cube.cube.reshape(1, -1))
        
        value = value[0, 0]

        return value, self.cube.is_solved()

    def search(self, network):
        if self.is_leaf():
            value, solved = self.expand(network)
            return value, solved
        else:
            action = self.policy()
            print(action)
            value, solved = self.children[action].search(network)
            self.w_s[action] = np.max([value, self.w_s[action]])
            self.n_s[action] += 1
            self.l_s[action] -= self.v
            return value, solved

class MCTS(object):
    def __init__(self, checkpoint, exploration = 2.0, virt_loss = 2.0):
        self.network = ADINetwork(output_dir = checkpoint)
        self.network.setup()
        self.max_steps = 200
        
        # Hyperparameters
        # Exploration parameter
        self.c = exploration
        self.v = virt_loss

    def search(self, cube):
        root_node = Node(None, cube, self.c, self.v, self.network)

        while True:
            _, solved= root_node.search(self.network)
            if solved:
                print("SOLVED PUZZLE YEEEEAAAAHHHHH")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Solve a single puzzle using MCTS")
    parser.add_argument("-c", "--checkpoint", required=True, help="tf checkpoint for network")
    parser.add_argument("-e", "--exploration", default = 0.01, type = float, help = "exploration hyperparameter")
    parser.add_argument("-v", "--v_loss", default = 0.01, type = float, help = "virtual loss hyperparameter")

    args = parser.parse_args()

    mcts = MCTS(args.checkpoint, args.exploration, args.v_loss)
    
    # Shuffle a cube a bunch
    cube = Cube()
    actions = np.eye(12)

    for i in range(5):
        cube.move(actions[np.random.choice(np.arange(0, actions.shape[0])), :])
    
    print("Starting search")
    mcts.search(cube)


