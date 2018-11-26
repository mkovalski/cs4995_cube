from rubiks import Cube
import pdb 
import numpy as np

#Class representing the game.
class Game_Rubiks :

  # Generate and return the initial game state. 
  def start(self):
    return Cube()
    

  # Return the current player's legal plays from given state. 
  def legalPlays(self, state):
    # if len(state.history) > 0:
    #   prev = state.history[-1]
    #   return [i for i in range(12) if not i == prev] 
    # else:
    return np.arange(12)

  # Advance the given state and return it. 
  def nextState(self, state, play) :
    new_state = Cube(state.scramble_distance, state.cube, state.history)
    new_state.rotate(play)

    return new_state

  #Return the winner of the game.
  def winner(self, state):
    if state.is_solved():
      return 1
    else: 
     return 0    
