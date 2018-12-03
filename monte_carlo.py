from monte_carlo_node import MonteCarloNode
import time
import pdb
import random 
import math
import numpy as np
from random import shuffle 
from mpi4py import MPI

from network import ADINetwork

"""
 * Class representing the Monte Carlo search tree.
 * Handles the four MCTS steps: selection, expansion, simulation, backpropagation.
 * Handles best-move selection.
"""
class MonteCarlo:

  """
   * Create a Monte Carlo search tree.
   * @param {Game} game - The game to query regarding legal moves and state advancement.
   * @param {number} UCB1ExploreParam - The square of the bias parameter in the UCB1 algorithm; defaults to 2.
   """
  def __init__(self, game, checkpoint = None, exploration = 2.0, virt_loss = 2.0, UCB1ExploreParam = 2, comm = None) :
    self.game = game
    self.nodes = {}
    
    if checkpoint is None:
      raise Exception("No checkpoint provided")
    self.network = ADINetwork(output_dir = checkpoint)
    self.network.setup()
  
    # Hyperparameters
    # Exploration parameter
    MonteCarloNode.c = exploration
    MonteCarloNode.v = virt_loss
    MonteCarloNode.network = self.network

    self.UCB1ExploreParam = UCB1ExploreParam

    self.parallel = True
    self.comm = comm
    if comm == None or comm.size < 2:
      self.parallel = False



  """
   * If state does not exist, create dangling node.
   * @param {State} state - The state to make a dangling node for; its parent is set to null.
  """
  def makeNode(self, state) :
    if not state.hash() in self.nodes:
      unexpandedPlays = self.game.legalPlays(state)
      node = MonteCarloNode(None, None, state, unexpandedPlays)
      self.nodes[state.hash()] = node
  
  

  """
   * From given state, run as many simulations as possible until the time limit, building statistics.
   * @param {State} state - The state to run the search from.
   * @param {number} timeout - The time to run the simulations for, in seconds.
   * @return {Object} Search statistics.
   """
  def runSearch(self, state, overall_timeout = 7, simulation_steps = 20) :

    self.makeNode(state)

    draws = 0
    totalSims = 0
    
    #pdb.set_trace()
    start = time.time()
    end = start + overall_timeout

    while time.time() < end:

      node = self.select(state)
      winner = self.game.winner(node.state)
      values, null = self.network.evaluate(state.cube.reshape(1, -1))
      value = values[0]

      if node.isLeaf() == False and winner == False:
        node = self.expand(node)
        winner, value = self.simulate(node = node, max_steps = simulation_steps)
      
      self.backpropagate(node, winner, value)

      if winner == 0:
        draws+=1
      totalSims+=1
    

    return { "runtime": time.time()-start, "simulations": totalSims, "draws": draws }
  

  """
   * From the available statistics, calculate the best move from the given state.
   * @param {State} state - The state to get the best play from.
   * @param {string} policy - The selection policy for the "best" play.
   * @return {Play} The best play, according to the given policy.
  """
  def bestPlay(self, state, policy = "robust"):

    self.makeNode(state)

    # If not all children are expanded, not enough information
    if self.nodes[state.hash()].isFullyExpanded() == False:
      #raise Exception("Not enough information!")
      return -1

    node = self.nodes[state.hash()]
    allPlays = node.allPlays()
    bestPlay = 0

    maxx = -np.Infinity

    # Most visits (robust child)
    if policy == "robust":
      for play in allPlays:
        childNode = node.childNode(play)
        if childNode.n_plays > maxx:
          bestPlay = play
          maxx = childNode.n_plays
        

    # Highest winrate (max child)
    elif policy == "max":
      for play in allPlays:
        childNode = node.childNode(play)
        ratio = childNode.n_wins / childNode.n_plays
        if ratio > maxx:
          bestPlay = play
          maxx = ratio

    if maxx == 0:
      print("Guessing")
      maxx = -np.Infinity
      bestPlay = np.argmax(node.p_s)

    return bestPlay
  

  """
   * Phase 1: Selection
   * Select until EITHER not fully expanded OR leaf node
   * @param {State} state - The root state to start selection from.
   * @return {MonteCarloNode} The selected node.
  """
  def select(self, state):
    node = self.nodes[state.hash()]
    while node.isFullyExpanded() and not node.isLeaf():
      plays = node.allPlays()
      bestPlay = 0
      bestUCB1 = -np.Infinity
      for play in plays:
        childUCB1 = node.childNode(play).getUCB1(self.UCB1ExploreParam)
        values, action = node.childNode(play).policy()
        childUCB1 = values[action]

        #pdb.set_trace()
        if childUCB1 > bestUCB1:
          bestPlay = play
          bestUCB1 = childUCB1
        
      
      node = node.childNode(bestPlay)
    
    return node
  

  """
   * Phase 2: Expansion
   * Of the given node, expand a random unexpanded child node
   * @param {MonteCarloNode} node - The node to expand from. Assume not leaf.
   * @return {MonteCarloNode} The new expanded child node.
  """
  def expand(self, node):
    plays = node.unexpandedPlays()

    null, policy= self.network.evaluate(node.state.cube.reshape(1, -1))
    policy= policy[0]

    reweighted_policy = []
    for p in plays: 
        reweighted_policy.append(policy[p[0]]*(random.random()))

    index = np.argmax(reweighted_policy)

    # if random.random() > .5:
    #   index = math.floor(random.random() * len(plays))

    play = plays[index][0]

    childState = self.game.nextState(node.state, play)
    childUnexpandedPlays = self.game.legalPlays(childState)
    childNode = node.expand(play, childState, childUnexpandedPlays)
    self.nodes[childState.hash()] = childNode

    return childNode
  

  """
   * Phase 3: Simulation
   * From given node, play the game until a terminal state, then return winner
   * @param {MonteCarloNode} node - The node to simulate from.
   * @return {number} The winner of the terminal game state.
  """
  def simulate(self, node = None, seed_state = None, max_steps=20):

    if not node is None:
      state = node.state
    elif seed_state is not None:
      state = seed_state
    else:
      raise Exception("node and seed_state cannot both be None in simulation")

    winner = self.game.winner(state)
    value, null = self.network.evaluate(state.cube.reshape(1, -1))
    value = value[0]

    if self.parallel and self.comm.Get_rank() == 0:
      # if winner:
      #   return winner, self.network.evaluate(state.cube.reshape(1, -1))

      plays = self.game.legalPlays(state)
      shuffle(plays)

      mpi_size = self.comm.Get_size()

      play = np.zeros(1, dtype=np.int32)
      for r in range(1, mpi_size):
        play[0] = plays[(r-1)%len(plays)]
        self.comm.Send([play, MPI.INT] , dest=r)

      self.comm.Bcast(node.state.cube)
      history_length = np.asarray([len(node.state.history)], dtype=np.int32)
      self.comm.Bcast(history_length)
      state_history = np.asarray(node.state.history, dtype=np.int32)
      self.comm.Bcast(state_history)

      winner = np.zeros(1, dtype=np.int32)
      winners = np.zeros(1, dtype = np.int32)
      self.comm.Reduce([winner, MPI.INT], [winners, MPI.INT], op=MPI.SUM, root = 0)

      value = np.zeros(1, dtype=np.int32)
      value[0] = -2147483648
      values = np.zeros(1, dtype = np.int32)
      self.comm.Reduce([value, MPI.INT], [values, MPI.INT], op=MPI.MAX, root = 0)

      if winners[0] > 0:
        print("0: !!!", winners)
        return winners[0], values[0]
      else: 
        return 0, values[0]

    else:
      while winner == False and len(state.history) < max_steps:
        plays = self.game.legalPlays(state)
        play = plays[math.floor(random.random() * len(plays))]
        state = self.game.nextState(state, play)
        winner = self.game.winner(state)

        value, null= self.network.evaluate(state.cube.reshape(1, -1))
        value = value[0]

        if winner > 0:
          print("!!!", winner)
     
      #pdb.set_trace()
      return winner, value
  

  """
   * Phase 4: Backpropagation
   * From given node, propagate plays and winner to ancestors' statistics
   * @param {MonteCarloNode} node - The node to backpropagate from. Typically leaf.
   * @param {number} winner - The winner to propagate.
  """
  def backpropagate(self, node, winner, value) :
    #pdb.set_trace()

    while not node == None:
      node.n_plays += 1
      # Parent's choice
      if winner == 1 :
        node.n_wins += 1

      action = np.argmax(node.p_s)
      node.w_s[action] = np.max([value, node.w_s[action]])
      node.n_s[action] += 1
      node.l_s[action] -= MonteCarloNode.v
      
      node = node.parent
    
