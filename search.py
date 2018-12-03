#!/usr/bin/env python

import numpy as np
from rubiks import Cube3x3
from network import ADINetwork
import tensorflow as tf
import copy
import argparse
import pdb
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Solve a single puzzle using MCTS")
    parser.add_argument("-c", "--checkpoint", required=True, help="tf checkpoint for network")

    args = parser.parse_args()
    
    network = ADINetwork(output_dir = args.checkpoint, use_gpu = False)
    network.setup()
    # Shuffle a cube a bunch
    cube = Cube3x3()
    actions = np.eye(12)
    
    total = 1000
    c = 0
    
    moves = 3
    for i in range(total):
        cube.reset()

        while cube.is_solved():
            cube.reset()
            for i in range(moves):
                cube.move(actions[np.random.choice(np.arange(0, actions.shape[0])), :])
        
        for i in range(moves*2):
            v, p = network.evaluate(cube.cube.reshape(1, -1))
            #print(np.argmax(p))
            solved =  cube.move(p)
            if solved == 1:
                c += 1
                break
    print(c/float(total))
