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
from  mpi4py import MPI
import random
import math

np.random.seed(seed=1)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

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

        self.n_plays = 0
        self.n_wins = 0

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

    def simulate(self):
        if not self.parallel:
            actions = np.eye(12)
            solved = False
            end = time.time() + self.simulation_time
            while not solved and time.time() < end:
                tmp_cube = Cube2x2(cube = self.cube.cube)
                solved = tmp_cube.is_solved()
                moves = 0
                while not solved and moves < cube.gods_number:
                    play = actions[math.floor(random.random() * len(actions))]
                    tmp_cube.move(play)
                    solved = tmp_cube.is_solved()
                    moves+=1

            if solved :
                return 1
            else:
                return 0
        else:
            sim = np.ones(1, dtype=np.int32)
            self.comm.Bcast(sim)
            self.comm.Bcast(self.cube.cube)
            winner = np.zeros(1, dtype=np.int32)
            winners = np.zeros(1, dtype = np.int32)
            self.comm.Reduce([winner, MPI.INT], [winners, MPI.INT], op=MPI.SUM, root = 0)

            return winners[0]

    def search(self, network, move_num = 0):
        # pdb.set_trace()
        if self.is_leaf():
            value, solved = self.expand(network)
            if solved:
                return value, solved, move_num, 1
            else:
                wins_detected= self.simulate()
                return value, solved, move_num, wins_detected
        else:
            action = self.policy()

            if self.run_simulation:
                if random.random() > 0.5:
                    action = np.random.choice(np.arange(0, 12))

                best_ratio =0
                for a, child in enumerate(self.children):
                    ratio = child.n_wins / child.n_plays if child.n_plays > 0 else 0
                    if ratio > best_ratio:
                        best_action = a
                        best_ratio = ratio

                # pdb.set_trace()
                if best_ratio > 0:
                    action = best_action

            value, solved, mn, wins_detected = self.children[action].search(network, move_num = move_num + 1)
            self.w_s[action] = np.max([value, self.w_s[action]])
            self.n_s[action] += 1
            self.l_s[action] -= self.v

            self.n_plays += 1
            if wins_detected >0:
                self.n_wins +=1

            return value, solved, mn, wins_detected

class MCTS(object):
    def __init__(self, checkpoint, cube, exploration = 0.1, virt_loss = 0.1, run_simulation =False, simulation_time = .5, parallel=False, comm=None):
        self.network = ADINetwork(output_dir = checkpoint, use_gpu = False)
        self.network.setup(cube.cube.size)
        self.max_steps = 200
        
        self.parallel = parallel
        Node.parallel = self.parallel
        if self.parallel:
            if comm is None:
                raise Exception("Cannot run in parallel without MPI comm")
            self.comm=comm
            Node.comm = self.comm
        self.simulation_time =simulation_time
        Node.simulation_time = self.simulation_time
        self.run_simulation = run_simulation
        Node.run_simulation = self.run_simulation

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
            _, solved, move_steps, _= root_node.search(self.network, move_num = 0)
            if solved:
                break
            num_moves += 1

        return solved, num_moves, move_steps

    def simulate(self, cube):
        actions = np.eye(12)
        end = time.time() + self.simulation_time
        solved = False

        while not solved and time.time() < end:
            tmp_cube = Cube2x2(cube = cube)
            solved = tmp_cube.is_solved()
            moves = 0
            while not solved and moves < tmp_cube.gods_number:
                play = actions[math.floor(random.random() * len(actions))]
                tmp_cube.move(play)
                solved = tmp_cube.is_solved()
                moves+=1
            if solved:
                break

        if solved:
            return 1
        else:
            return 0

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    parallel = True if mpi_size > 1 else False

    parser = argparse.ArgumentParser("Solve a single puzzle using MCTS")
    parser.add_argument("-c", "--checkpoint", required=True, help="tf checkpoint for network")
    parser.add_argument("-e", "--exploration", default = 0.1, type = float, help = "exploration hyperparameter")
    parser.add_argument("-v", "--v_loss", default = 0.000001, type = float, help = "virtual loss hyperparameter")
    parser.add_argument("-d", "--dims", default = 2, choices = [2, 3], type = int, help = "Cube dimensions")
    parser.add_argument("-b", "--batch", default = 1, type = int)
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--run_mcts_simulation", action = "store_true", help="run mini simulations to search for wins from leaf nodes, changes search algorithm")
    parser.add_argument("--simulation_time", type = float, default = .5, help="time to run mini simulations ")
    parser.add_argument("-m", "--moves", type = int, default = 5)
    parser.add_argument("-t", "--time", type = int, help = "Time limit for overall MCTS", default = 60)

    args = parser.parse_args()
    
    # Shuffle a cube a bunch
    if args.dims == 2:
        cube = Cube2x2()
    else:
        cube = Cube3x3()
    
    if parallel and not args.run_mcts_simulation:
        raise Exception("Cannot utilize parallelization without running MCTS simulations.  Add arg: --run_mcts_simulation")

    mcts = MCTS(args.checkpoint, cube, args.exploration, args.v_loss, run_simulation = args.run_mcts_simulation, simulation_time = args.simulation_time, parallel = parallel, comm = comm)
    
    if mpi_rank ==0:
        actions = np.eye(12)
        
        num_solved = 0
        total_cubes = args.batch

        nodes_explored = []
        solve_length = []

        for b in range(args.batch):
            cube.reset()
            win_moves = []
            for i in range(args.moves):
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
        

        print("Statistics: solving {} moves away".format(args.moves))
        print(" - Percent solved: {}".format(num_solved / float(total_cubes)))
        print(" - Average num nodes explored: {}".format(np.mean(nodes_explored)))
        print(" - Average number of steps to solve length: {}".format(np.mean(solve_length)))

        sim = np.zeros(1, dtype=np.int32)
        comm.Bcast(sim)
    else:
        sim = np.ones(1, dtype=np.int32)
        cube_rep = np.zeros(cube.cube.shape, dtype=np.int32)
        wins_detected = np.zeros(1, dtype=np.int32)
        null = np.zeros(1, dtype = np.int32)
        while True:
            comm.Bcast(sim, root = 0)
            if sim[0] == 0:
                break
            comm.Bcast(cube_rep)
            wins_detected[0] = mcts.simulate(cube_rep)
            comm.Reduce([wins_detected, MPI.INT], [null, MPI.INT], op=MPI.SUM, root = 0)

