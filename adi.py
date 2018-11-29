#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from network import ADINetwork
from rubiks import Cube
import argparse
import copy
import pdb

def move(cube, depth):
    if depth == 0:
        return cube.is_solved()
    else:
        cubes = np.zeros((12, 24*20))
        true_values = np.zeros((12, 1))
        

def adi(M = 2000000, L = 10, steps_per_iter = 2000, allow_move_back = False, batch = False,
        search_depth = 1, output_dir = 'output'):

    '''Function for Autodidactic Iteration
    Start at the initial state each run, take N random actions sequentially,
    and train the value policy network on the case of examples
    
    '''
    
    # Setup cube
    cube = Cube()
    
    # The maximum that the batch size can be
    # since it won't fit into gpu memory otherwise
    max_batch = 400
    
    # Start with moves very close to the cube
    # after some time, increase
    K = 1

    # Set up the neural network
    network = ADINetwork(output_dir)
    network.setup()
    
    # Moves allowed to make
    actions = np.eye(12).astype(np.float32)

    # Number of times to run each of the simulations at each
    # value of K
    steps = np.arange(1, 31) * steps_per_iter

    # Run a ton at the end
    add_steps = (M - np.sum(steps))
    if add_steps > 0:
        steps[-1] += add_steps

    print("Running {} total steps".format(np.sum(steps)))
    
    local_steps = 0
    step_idx = 0

    for m in range(1, M+1):
        N = K * L

        all_values = np.zeros((N, 1))
        all_policies = np.zeros((N, 12))
        all_states = np.zeros((N, 24 * 20))

        weight_vector = np.ones(N)
        
        for l in range(L):
            cube.reset()

            last_moves = np.asarray([]).astype(int)
            w = 0

            for k in range(K):

                true_values = np.zeros((actions.shape[0], 1))

                cubes = np.zeros((12, 24 * 20))
                
                # Iterate over all the actions from a given state
                # Store the true reward and the states
                for mv in range(actions.shape[0]):
                    tmp_cube = copy.deepcopy(cube)
                    true_values[mv, :] = tmp_cube.move(actions[mv, :])
                    cubes[mv, :] = tmp_cube.cube.reshape(1, -1)
                    del tmp_cube
                
                vals, _ = network.evaluate(cubes)
                vals += true_values
                
                idx = np.argmax(vals)
                all_values[(l*K) + k, :] = vals[idx]
                all_policies[(l*K) + k, :] = actions[idx, :]
                all_states[(l*K) + k, :] = copy.copy(cube.cube).flatten()

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
                max_action = np.argmax(action)

                # Adjust weight vector here
                '''
                if last_moves.size > 0:
                    ind = 1 if last_moves[0] % 2 == 0 else -1
                    if last_moves[0] + ind == max_action:
                        w -= 1
                    elif last_moves.size >= 2 and np.all(last_moves[-2:] == max_action):
                        w -= 1
                    else:
                        w += 1
                else:
                    w += 1
                '''

                w += 1

                # Queueing
                last_moves = np.insert(last_moves, 0, max_action)

                if last_moves.size > 3:
                    last_moves = np.delete(last_moves, -1)
                cube.move(action)

                weight_vector[(l*K) + k] = 1 / w
        
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

        local_steps += 1
        if local_steps >= steps[step_idx]:
            step_idx += 1
            local_steps = 0
            K += 1
            print("-- Increasing K to {}".format(K))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rubik's cude using autodidactic iteration")
    parser.add_argument("-M", type = int, default = 2000000, help = "Number of trials")
    parser.add_argument("-L", type = int, default = 40, help = "How many moves to make from state to state")
    parser.add_argument("--steps_per_iter", type = int, default = 500, help = "How many moves to make from state to state")
    parser.add_argument('--allow_move_back', action='store_true', help = "Allow the rubik's cube to move to it's previous state during the search")
    parser.add_argument('--batch', action='store_true', help="Train the neural network in batches")
    parser.add_argument('-o', '--output_dir', type = str, default = 'output', help="Where to save tensorflow checkpoints to")

    args = parser.parse_args()
    adi(M = args.M, L = args.L, steps_per_iter = args.steps_per_iter,
        allow_move_back = args.allow_move_back, batch = args.batch,
        output_dir = args.output_dir)
