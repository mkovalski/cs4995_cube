import math
import numpy as np

"""
 * Class representing a node in the search tree.
 * Stores tree search stats for UCB1.
 """
class MonteCarloNode :
  """
   * Create a new MonteCarloNode in the search tree.
   * @param {MonteCarloNode} parent - The parent node.
   * @param {Play} play - Last play played to get to this state.
   * @param {State} state - The corresponding state.
   * @param {Play[]} unexpandedPlays - The node's unexpanded child plays.
  """
  def __init__(self, parent, play, state, unexpandedPlays) :

    self.num_actions = 12


    self.play = play
    self.state = state

    # Monte Carlo stuff
    self.n_plays = 0
    self.n_wins = 0

    # Tree stuff
    self.parent = parent
    self.children = {}
    for play in unexpandedPlays :
      self.children[play] = { "play": play, "node": None }
    
  

  """
   * Get the MonteCarloNode corresponding to the given play.
   * @param {number} play - The play leading to the child node.
   * @return {MonteCarloNode} The child node corresponding to the play given.
  """
  def childNode(self, play) :
    child = self.children[play]
    if child == None:
      raise Exception( 'No such play!')
    
    elif child["node"] == None:
      raise Exception( "Child is not expanded!")
    
    return child["node"]
  

  """
   * Expand the specified child play and return the new child node.
   * Add the node to the array of children nodes.
   * Remove the play from the array of unexpanded plays.
   * @param {Play} play - The play to expand.
   * @param {State} childState - The child state corresponding to the given play.
   * @param {Play[]} unexpandedPlays - The given child's unexpanded child plays; typically all of them.
   * @return {MonteCarloNode} The new child node.
  """
  def expand(self, play, childState, unexpandedPlays) :
    if not play in self.children:
      raise Exception( "No such play!")
    childNode = MonteCarloNode(self, play, childState, unexpandedPlays)
    self.children[play]=  { "play": play, "node": childNode }
    return childNode
  

  """
   * Get all legal plays from this node.
   * @return {Play[]} All plays.
  """
  def allPlays(self) :
    ret = []
    for k, child in self.children.items() :
      ret.append(child["play"])
    
    return ret
  

  """
   * Get all unexpanded legal plays from this node.
   * @return {Play[]} All unexpanded plays.
  """
  def unexpandedPlays(self) :
    ret = []
    for k, child in self.children.items() :
      if child["node"] == None:
        ret.append(child["play"])
    
    return ret
  

  """
   * Whether this node is fully expanded.
   * @return {boolean} Whether this node is fully expanded.
  """
  def isFullyExpanded(self) :
    for k, child in self.children.items():
      if child["node"] == None :
        return False
    
    return True
  

  """
   * Whether this node is terminal in the game tree, NOT INCLUSIVE of termination due to winning.
   * @return {boolean} Whether this node is a leaf in the tree.
  """
  def isLeaf(self) :
    if len(self.children) == 0:
      return True
    else :
      return False
  
  
  """
   * Get the UCB1 value for this node.
   * @param {number} biasParam - The square of the bias parameter in the UCB1 algorithm, defaults to 2.
   * @return {number} The UCB1 value of this node.
  """
  def getUCB1(self, biasParam) :
    return (self.n_wins / self.n_plays) + math.sqrt(biasParam * math.log(self.parent.n_plays) / self.n_plays)
  

  def UCT(self):
    return (self.c * self.p_s * np.sqrt(np.sum(self.n_s)))/(1 + self.n_s)
    
  def QCT(self):
    q = self.w_s - self.l_s
    self.l_s += self.v
    return q

  def policy(self):
    return np.argmax(self.UCT() + self.QCT())
