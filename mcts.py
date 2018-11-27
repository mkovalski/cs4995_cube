#!/usr/bin/env python

import numpy as np
from rubiks import Cube
from network import ADINetwork
import tensorflow as tf
import copy
import argparse
import pdb
import sys

from game import Game_Rubiks
from monte_carlo import MonteCarlo

from mpi4py import MPI

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


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    parser = argparse.ArgumentParser("Solve a single puzzle using MCTS")
    #parser.add_argument("-c", "--checkpoint", required=True, help="tf checkpoint for network")
    parser.add_argument("-e", "--exploration", default = 0.01, type = float, help = "exploration hyperparameter")
    parser.add_argument("-v", "--v_loss", default = 0.01, type = float, help = "virtual loss hyperparameter")
    parser.add_argument("-t", "--simulation-time", default=.5, type=float, help="time limit per simulation")
    parser.add_argument("-o", "--overall-time", default=0, type=float, help="time limit tree search")
    parser.add_argument("-s", "--shuffle", default=3, type=int, help="shuffles from the start state")


    args = parser.parse_args()
    
    game = Game_Rubiks()
    mcts = MonteCarlo(game, comm = comm)

    if mpi_rank == 0:
        parallel = True if mpi_size > 1 else False
        if parallel:
            print("Running in parallel")

        # Shuffle a cube a bunch
        cube = Cube()
        actions = np.eye(12)

        for i in range(args.shuffle):
            cube.move(actions[np.random.choice(np.arange(0, actions.shape[0])), :])
        
        if args.overall_time < args.simulation_time:
            args.overall_time = args.simulation_time + 1


        print(cube.history)
    #####

        state = cube
        winner= False
        i = 0
        retry = 0
        stats = {"runtime":0}
        while winner == False:
            while retry < 5:
                print(i, retry)
                stats = mcts.runSearch(state, overall_timeout=args.overall_time*(retry+1), simulation_timeout = args.simulation_time)
                play = mcts.bestPlay(state, policy="max")
                if play >= 0:
                    break

                retry+=1

            retry = 0
            if play < 0:
                print("Not enough information!")
                pdb.set_trace()

            print("   {} {}".format(play, stats["runtime"]))
            state = game.nextState(state, play)
            winner = game.winner(state)
            i += 1

            if i % 5 ==0 :
                pdb.set_trace()

        print("YOU WON", state.history)

        finished = np.asarray([-1], dtype=np.int32)
        for r in range(1, mpi_size):
            comm.Send([finished, MPI.INT], dest= r)

    else:
        play = np.zeros(1, dtype=np.int32)
        cube_rep = np.zeros((20, 24), dtype=np.int32)
        winner = np.zeros(1, dtype=np.int32)
        null = np.zeros(1, dtype = np.int32)

        while True:
            comm.Recv([play, MPI.INT], source=0)

            if play[0] < 0:
                break
            else:
                comm.Bcast(cube_rep, root=0)

                state = Cube(cube = cube_rep)

                state = game.nextState(state, play)
                winner[0] = mcts.simulate(seed_state = state, timeout = args.simulation_time)

                comm.Reduce([winner, MPI.INT], [null, MPI.INT], op=MPI.SUM, root = 0)




