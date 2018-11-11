# cs4995_cube

## How to run
There are two parts to the code base, the autodidactic iteration and the monte carlo tree search.

### Autodidactic Iteration
To run the initial training of the neural network, one would you the script `adi.py`. To see a full list of options for training, run `python adi.py -h`

```
usage: Rubik's cude using autodidactic iteration [-h] [-m M] [-n N]
                                                 [--allow_move_back] [--batch]
                                                 [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -m M                  Number of trials
  -n N                  How many moves to make from state to state
  --allow_move_back     Allow the rubik's cube to move to it's previous state
                        during the search
  --batch               Train the neural network in batches
  --output_dir OUTPUT_DIR
                        Where to save tensorflow checkpoints to

```

### Monte Carlo Tree Search
Once the network is trained, we use this network to guide the tree search. To see the list of options, run `python mcts.py -h`


```
usage: Solve a single puzzle using MCTS [-h] -c CHECKPOINT [-e EXPLORATION]
                                        [-v V_LOSS]

optional arguments:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        tf checkpoint for network
  -e EXPLORATION, --exploration EXPLORATION
                        exploration hyperparameter
  -v V_LOSS, --v_loss V_LOSS
                        virtual loss hyperparameter

```

# Project overview

See proposal_README.pdf
