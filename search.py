#!/usr/bin/env python

import numpy as np
from rubiks import Cube
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
    
    network = ADINetwork(output_dir = args.checkpoint)
    network.setup()
    # Shuffle a cube a bunch
    cube = Cube()
    actions = np.eye(12)

    for i in range(1):
        cube.move(actions[np.random.choice(np.arange(0, actions.shape[0])), :])

    print("Starting search")
    
    while True:
        v, p = network.evaluate(cube.cube.reshape(1, -1))
        print(np.argmax(p))
        pdb.set_trace()
        solved =  cube.move(p)
        if solved == 1:
            break

