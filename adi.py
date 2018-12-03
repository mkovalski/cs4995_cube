#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from network import ADINetwork
from rubiks import Cube2x2, Cube3x3
import argparse
import copy
import pdb
import sys
import time

def move(cube, depth):
    if depth == 0:
        return cube.is_solved()
    else:
        cubes = np.zeros((12, 24*20))
        true_values = np.zeros((12, 1))
        

def adi(M = 2000000, max_K = 30, L = 10, steps_per_iter = 2000, gamma = 0.5, 
        allow_move_back = False, search_depth = 1, output_dir = 'output', 
        same_batch = False, use_cpu = False, dims = 3):

    '''Function for Autodidactic Iteration
    Start at the initial state each run, take N random actions sequentially,
    and train the value policy network on the case of examples
    
    '''
    
    # Setup cube
    if dims == 2:
        cube = Cube2x2()
    else:
        cube = Cube3x3()
    
    # Start with moves very close to the cube
    # after some time, increase
    K = 1
    orig_L = copy.copy(L)

    # Set up the neural network
    network = ADINetwork(output_dir, use_gpu = not use_cpu)
    network.setup(cube_size = cube.cube.size)
    
    # Moves allowed to make
    actions = np.eye(12).astype(np.float32)

    # Number of times to run each of the simulations at each
    # value of K. Scale linearly
    steps = np.arange(1, max_K+1) * steps_per_iter

    # Run a ton at the end if there are leftover steps
    add_steps = (M - np.sum(steps))
    if add_steps > 0:
        steps[-1] += add_steps

    print("Running {} total steps".format(np.sum(steps)))
    
    local_steps = 0
    step_idx = 0
    
    print("Using batch size of {}".format(L))
    
    for m in range(1, M+1):
        N = K * L

        all_values = np.zeros((N, 1))
        all_policies = np.zeros((N, 12))
        all_states = np.zeros((N, cube.cube.size))
        
        weight_vector = np.ones(N)
                
        cubes = np.zeros((N * 12, cube.cube.size))
        true_values = np.zeros((N * 12, 1))
        
        for l in range(L):
            cube.reset()

            last_moves = np.asarray([]).astype(int)
            w = 1

            for k in range(K):
                
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
                
                help_idx = max_action - 1 if max_action%2 ==1 else max_action + 1
                
                # Queueing
                last_moves = np.insert(last_moves, 0, max_action)

                if last_moves.size > 3:
                    last_moves = np.delete(last_moves, -1)

                cube.move(action)

                weight_vector[(l*K) + k] = 1 / w
                
                w += 1

                # Iterate over all the actions from a given state
                # Store the true reward and the states
                for mv in range(actions.shape[0]):
                    idx = (l * K * 12) + (k*12) + mv
                    #print(idx)
                    tmp_cube = copy.deepcopy(cube)
                    true_values[idx, :] = tmp_cube.move(actions[mv, :])
                    cubes[idx, :] = tmp_cube.cube.reshape(1, -1)
                    del tmp_cube
                
                all_states[(l*K) + k, :] = np.copy(cube.cube).flatten()
        
        vals, _ = network.evaluate(cubes)
        vals *= gamma
        vals += true_values

        # Ignore the discount factor when we know we reached a terminal state
        # We just want 1 here
        vals[np.where(true_values == 1)[0]] = 1

        vals = vals.reshape(-1, 12)
        idx = np.argmax(vals, axis = 1)
        all_values[:, :] = np.max(vals, axis = 1).reshape(-1, 1)
        all_policies[:, :] = actions[idx, :]
        
        cost = network.train(all_states, all_policies, all_values,
                             weight = weight_vector)

        if m % 10 == 0:
            print("Iteration {} complete".format(m))

        if m % 10 == 0:
            print("- Latest cost: {}".format(cost))
            sys.stdout.flush()

        # Log stuff
        if m % 10 == 0:
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
            if same_batch:
                L = orig_L // K
                if L == 0:
                    L = 1
                print("L is now {}".format(L))

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rubik's cude using autodidactic iteration")
    parser.add_argument("-M", type = int, default = 2000000, help = "Number of trials overall")
    parser.add_argument("-K", type = int, default = 30, help = "Maximum moves away it will move from solved state")
    parser.add_argument("-L", type = int, default = 50, help = "Number of times to run 1:K before sending as batch to NN. K * L gives batch size unless --same_batch is specified as argument")
    parser.add_argument("-g", "--gamma", type = float, default = 0.5, help = "Discount factor")
    parser.add_argument("--steps_per_iter", type = int, default = 500, help = "How many moves to make at each K. Linear function, so number moves at each k is K * steps_per_iter")
    parser.add_argument("--same_batch", action = 'store_true', help="Rescale batch as K grows")
    parser.add_argument('--allow_move_back', action='store_true', help = "Allow the rubik's cube to move to it's previous state during the search")
    parser.add_argument('-o', '--output_dir', type = str, default = 'output', help="Where to save tensorflow checkpoints to")
    parser.add_argument('--use_cpu', action = 'store_true')
    parser.add_argument('-d', '--dims', type = int, choices=[2, 3], help = 'Cube dimensions')

    args = parser.parse_args()

    adi(M = args.M, max_K = args.K, L = args.L, steps_per_iter = args.steps_per_iter,
        gamma = args.gamma, allow_move_back = args.allow_move_back,
        output_dir = args.output_dir, same_batch = args.same_batch,
        use_cpu = args.use_cpu, dims = args.dims)
