from monte_carlo_node import MonteCarloNode
import time
import pdb
import random 
import math
import numpy as np

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
  def __init__(self, game, checkpoint = None, exploration = 2.0, virt_loss = 2.0, UCB1ExploreParam = 2) :
    self.game = game
    self.nodes = {}
    
    # self.network = ADINetwork(output_dir = checkpoint)
    # self.network.setup()
  
    # Hyperparameters
    # Exploration parameter
    MonteCarloNode.c = exploration
    MonteCarloNode.v = virt_loss

    self.UCB1ExploreParam = UCB1ExploreParam


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
  def runSearch(self, state, overall_timeout = 7, simulation_timeout = .4) :

    self.makeNode(state)

    draws = 0
    totalSims = 0
    
    end = time.time() + overall_timeout

    while time.time() < end:

      node = self.select(state)
      winner = self.game.winner(node.state)

      if node.isLeaf() == False and winner == False:
        node = self.expand(node)
        winner = self.simulate(node, simulation_timeout)
      
      self.backpropagate(node, winner)

      if winner == 0:
        draws+=1
      totalSims+=1
    

    return { "runtime": overall_timeout, "simulations": totalSims, "draws": draws }
  

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

    else:
      for play in allPlays:
        childNode = node.childNode(play)
        if childNode.value > maxx:
          bestPlay = play
          maxx = childNode.value 

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
        #_, childUCB1 = node.childNode(play).policy()
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
    index = math.floor(random.random() * len(plays))
    play = plays[index]

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
  def simulate(self, node, timeout):

    state = node.state
    winner = self.game.winner(state)
    end = time.time()+timeout

    while winner == False and time.time()< end:
      plays = self.game.legalPlays(state)
      play = plays[math.floor(random.random() * len(plays))]
      state = self.game.nextState(state, play)
      winner = self.game.winner(state)
    

    return winner
  

  """
   * Phase 4: Backpropagation
   * From given node, propagate plays and winner to ancestors' statistics
   * @param {MonteCarloNode} node - The node to backpropagate from. Typically leaf.
   * @param {number} winner - The winner to propagate.
  """
  def backpropagate(self, node, winner) :

    while not node == None:
      node.n_plays += 1
      # Parent's choice
      if winner == 1 :
        node.n_wins += 1

      # node.w_s[action] = np.max([value, node.w_s[action]])
      # node.n_s[action] += 1
      # node.l_s[action] -= self.v
      
      node = node.parent
    
