#!/usr/bin/env python

import numpy as np
from rubiks import Cube2x2, Cube3x3
from network import ADINetwork
import tensorflow as tf
import copy
import argparse
import pdb
import sys
import time

class Node(object):
    ''' A node in the MCTS algorithm. Contains information such as number
    of visits, etc.
    
    '''
    def __init__(self, parent, cube, exploration, virt_loss, network):
        
        self.num_actions = 12

        self.parent = parent
        self.cube = cube
        self.c = exploration
        self.v = virt_loss

        self.tol = 0.00001
        
        _, self.p_s = network.evaluate(self.cube.cube.reshape(1, -1))

        # Number of times an action a has been taken from state
        self.n_s = np.zeros(self.num_actions)

        # Maximal value of action a from state s
        self.w_s = np.zeros(self.num_actions)
        
        # The current virtual loss for action a from state s 
        self.l_s = np.zeros(self.num_actions)

        self.children = np.asarray([None] * self.num_actions)
        
    def UCT(self):
        return (self.c * self.p_s * np.sqrt(np.sum(self.n_s) + self.tol))/(1 + self.n_s)
    
    def QCT(self):
        q = self.w_s - self.l_s
        self.l_s += self.v
        return q

    def policy(self):
        pol = self.UCT() + self.QCT()
        if np.unique(pol).size == 1:
            return np.random.choice(np.arange(0, self.num_actions))
        else:
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

    def search(self, network, move_num = 0):
        if self.is_leaf():
            value, solved = self.expand(network)
            return value, solved, move_num
        else:
            action = self.policy()
            value, solved, mn = self.children[action].search(network, move_num = move_num + 1)
            self.w_s[action] = np.max([value, self.w_s[action]])
            self.n_s[action] += 1
            self.l_s[action] -= self.v
            return value, solved, mn

class MCTS(object):
    def __init__(self, checkpoint, cube, exploration = 0.1, virt_loss = 0.1):
        self.network = ADINetwork(output_dir = checkpoint, use_gpu = False)
        self.network.setup(cube.cube.size)
        self.max_steps = 200
        
        # Hyperparameters
        # Exploration parameter
        self.c = exploration
        self.v = virt_loss

    def search(self, cube, time_limit = 60):
        
        root_node = Node(None, cube, self.c, self.v, self.network)
        
        num_moves = 0
        t1 = time.time()
       
        solved = False
        while(time.time() - t1 < time_limit):
            _, solved, move_steps= root_node.search(self.network, move_num = 0)
            if solved:
                break
            num_moves += 1

        return solved, num_moves, move_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Solve a single puzzle using MCTS")
    parser.add_argument("-c", "--checkpoint", required=True, help="tf checkpoint for network")
    parser.add_argument("-e", "--exploration", default = 0.1, type = float, help = "exploration hyperparameter")
    parser.add_argument("-v", "--v_loss", default = 0.000001, type = float, help = "virtual loss hyperparameter")
    parser.add_argument("-d", "--dims", default = 2, choices = [2, 3], type = int, help = "Cube dimensions")
    parser.add_argument("-b", "--batch", default = 1, type = int)
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("-m", "--moves", type = int, default = [5], nargs = '+')
    parser.add_argument("-t", "--time", type = int, help = "Time limit", default = 60)
    parser.add_argument("-p", "--plot", action = "store_true")

    args = parser.parse_args()
    
    # Shuffle a cube a bunch
    if args.dims == 2:
        cube = Cube2x2()
    else:
        cube = Cube3x3()
    
    mcts = MCTS(args.checkpoint, cube, args.exploration, args.v_loss)
    
    actions = np.eye(12)
    
    num_solved = 0
    total_cubes = args.batch

    
    for moves in args.moves:
        print("Running cubes {} away for {} iterations".format(moves, args.batch))
        
        nodes_explored = []
        solve_length = []
        num_solved = 0

        for b in range(args.batch):
            cube.reset()
            win_moves = []
            for i in range(moves):
                action = np.random.choice(np.arange(0, actions.shape[0]))
                cube.move(actions[action, :])
                flip = 1 if action % 2 == 0 else -1
                win_moves.append(action + flip)
            
            if args.verbose:
                for it in reversed(win_moves):
                    print(" - Take action {}".format(it))

                print("Starting search")

            solved, num_moves, move_steps = mcts.search(cube, time_limit = args.time)
            if solved:
                num_solved += 1
                nodes_explored.append(num_moves)
                solve_length.append(move_steps)
        
        print("Statistics: solving {} moves away".format(moves))
        print(" - Percent solved: {}".format(num_solved / float(total_cubes)))
        print(" - Average num nodes explored: {}".format(np.mean(nodes_explored)))
        print(" - Average number of steps to solve length: {}".format(np.mean(solve_length)))
        print()

