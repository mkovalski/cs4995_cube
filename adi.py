#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from network import ADINetwork
from rubiks import Cube
import argparse
import copy
import pdb

def adi(M = 1000, N = 1000, allow_move_back = False, batch = False,
        output_dir = 'output'):

    '''Function for Autodidactic Iteration
    Start at the initial state each run, take N random actions sequentially,
    and train the value policy network on the case of examples
    
    '''
    
    # Setup cube
    cube = Cube()

    # Set up the neural network
    network = ADINetwork(output_dir)
    network.setup(batch_size = N if batch else 1)
    
    # Moves allowed to make
    actions = np.eye(12).astype(np.float32)

    for m in range(1, M+1):
        cube.reset()

        last_moves = np.asarray([]).astype(int)

        all_values = np.zeros((N, 1))
        all_policies = np.zeros((N, 12))
        all_states = np.zeros((N, 24 * 20))
        
        w = 1
        weight_vector = np.ones((N, 1))

        for n in range(N):
            true_values = np.zeros((actions.shape[0], 1))

            cubes = np.zeros((12, 24 * 20))
            
            # Iterate over all the actions from a given state
            # Store the true reward and the states
            for k in range(actions.shape[0]):
                tmp_cube = copy.deepcopy(cube)
                true_values[k, :] = tmp_cube.move(actions[k, :])
                cubes[k, :] = tmp_cube.cube.reshape(1, -1)
                del tmp_cube
            
            vals, _ = network.evaluate(cubes)
            vals += true_values
            
            idx = np.argmax(vals)
            all_values[n, :] = vals[idx]
            all_policies[n, :] = actions[idx, :]
            all_states[n, :] = copy.copy(cube.cube).flatten()
            
            if not batch:
                cost = network.train(all_states[n, :].reshape(1, -1), 
                    all_policies[n, :].reshape(1, -1), 
                    all_values[n, :].reshape(1, -1))

            # Try some different stuff out
            # Don't move the cube back to a position that it was just in, need
            # to explore more
            choices = np.arange(0, actions.shape[0])

            if not allow_move_back:
                r_ind = []
                if last_moves.size != 0:
                    ind = 1 if last_moves[0] % 2 == 0 else -1
                    r_ind.append(last_moves[0] + ind)

                # In addition, if there are three of the same moves in a row,
                # don't allow same move
                if last_moves.size == 3 and np.unique(last_moves).size == 1:
                    r_ind.append(last_moves[0])

                if r_ind:
                    choices = np.delete(choices, r_ind)

            action = actions[np.random.choice(choices), :]
            
            # Adjust weight vector here
            if len(last_moves >= 2) and np.all(last_moves[-2:] == action):
                w -= 1
            else:
                w += 1
            
            # Queueing
            last_moves = np.insert(last_moves, 0, np.argmax(action))
            if last_moves.size > 3:
                last_moves = np.delete(last_moves, -1)
            cube.move(action)

            weight_vector[n] = 1 / w
        
        if batch:
            cost = network.train(all_states, all_policies, all_values,
                                 weight = weight_vector)

        print("Iteration {} complete".format(m))
        print("- Latest cost: {}".format(cost))
        
        # Log stuff
        if m % 10:
            network.log()
        
        # Checkpoint
        if m % 1000 == 0:
            print("-- Saving network at iteration {}".format(m))
            network.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rubik's cude using autodidactic iteration")
    parser.add_argument("-m", type = int, default = 2000000, help = "Number of trials")
    parser.add_argument("-n", type = int, default = 100, help = "How many moves to make from state to state")
    parser.add_argument('--allow_move_back', action='store_true', help = "Allow the rubik's cube to move to it's previous state during the search")
    parser.add_argument('--batch', action='store_true', help="Train the neural network in batches")
    parser.add_argument('-o', '--output_dir', type = str, default = 'output', help="Where to save tensorflow checkpoints to")

    args = parser.parse_args()
    adi(M = args.m, N = args.n, 
        allow_move_back = args.allow_move_back, batch = args.batch,
        output_dir = args.output_dir)
