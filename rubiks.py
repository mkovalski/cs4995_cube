#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:12:06 2018

@author: vsomdec
"""

import numpy as np
import random
import sys

from abc import ABC, abstractmethod

class Cube(ABC):
    
    #dominant orientations: Top(white), Bottom(yellow)
    #secondary: Left(blue), Right(green)

    #unique cube positions
    cublets = {}
    inv_cublets = {v: k for k, v in cublets.items()}

    #all states for edges
    edge_pos = {}
    inv_edge_pos = {v: k for k, v in edge_pos.items()}
           
    #all states for corners
    corner_pos = {}
    inv_corner_pos = {v: k for k, v in corner_pos.items()}

  
    #Begin with solved cube state or seed with cube state
    def __init__(self, scramble_distance=0, cube = None, history = None):
        if history is None:
            history = []
        if cube is None:
            self.reset()
        else :
            self.cube = np.copy(cube)
            self.scramble_distance = scramble_distance
            self.history = history[:]
    
    #Reset to solved cube state
    def reset(self):
        self.cube = self.solved_cube()
        self.scramble_distance = 0
        self.history = []
    
    def hash(self):
        return hash(self.cube.tostring()) + hash(np.asarray(self.history).tostring())

    #Get highest probability out of list of moves and rotate accordingly
    def move(self, move_prob):
        move = np.argmax(move_prob)
        self.rotate(move)
        return int(self.is_solved())
        
    #Swap positions of two corners/edge orientations
    def swap(self, a, b):
    #Get orientation in given positions
        a_ = np.argmax(self.cube[self.cublets[a]])
        b_ = np.argmax(self.cube[self.cublets[b]])
    #Reset row, only one orientation can exist in a given position
        self.cube[self.cublets[a]].fill(0)
        self.cube[self.cublets[b]].fill(0)
    #Set the orientation to true, swapping positions
        self.cube[self.cublets[a], b_] = 1 
        self.cube[self.cublets[b], a_] = 1  
    
    #Perform similar to swap function, however reverse edge orientation
    def swap_flip_edge(self, a, b):
        if len(a) > 2 or len(b) > 2:
            print("a:{} and b:{} must be an edge to flip".format(a, b))
            sys.exit(1)
            
        a_ = np.argmax(self.cube[self.cublets[a]])
        b_ = np.argmax(self.cube[self.cublets[b]])
    #Reverse orientation
        a_ = self.edge_pos[self.inv_edge_pos[a_][::-1]]
        b_ = self.edge_pos[self.inv_edge_pos[b_][::-1]]
        
        self.cube[self.cublets[a]].fill(0)
        self.cube[self.cublets[b]].fill(0)
        self.cube[self.cublets[a], b_] = 1 
        self.cube[self.cublets[b], a_] = 1  

    #Rotate orientation clockwise or counterclockwise in current position
    def reorient_corner(self, a, dire):
        cor = np.argmax(self.cube[self.cublets[a]])
        
        if dire == 1:
            cor = self.clockwise(self.inv_corner_pos[cor])
        elif dire == -1:
            cor = self.counterclockwise(self.inv_corner_pos[cor])
        else: 
            print("Can only turn clockwise(1), or anticlockwise(-1)")
            sys.exit(1)

        self.cube[self.cublets[a]].fill(0)
        self.cube[self.cublets[a], self.corner_pos[cor]] = 1
                     
        
    def clockwise(self, a):
        # WGO -> OWG
        return a[-1]+a[0:-1]
    
    def counterclockwise(self, a):
        # WGO -> GOW
        return a[1]+a[2]+a[0]
        
    #Returns true if state matches solved state
    def is_solved(self):
        return np.all(self.cube == self.solved_cube())
        
    #Perform rotation of given face 90 degrees clockwise
    #Note a counterclockwise move is 3 clockwise rotations 
    @abstractmethod
    def rotate(self, move):
        self.scramble_distance+=1
        self.history.append(move)
        
        pass

    #Solved state, each orientation is in default position
    @abstractmethod
    def solved_cube(self):
        solved = np.zeros((20, 24), dtype = np.int32)
        #return solved
        pass      

    def test(self, M = 100, N = 100):
        
        pos_moves = 12
        for n in range(N):
            self.reset()
            
            actions = np.eye(pos_moves)[np.random.choice(pos_moves, M)]

            for m in range(M):
                self.move(actions[m, :])        
            
            # Reverse moves
            rev_idx = np.where(actions == 1)[1]
            mod_moves = rev_idx % 2
            rev_idx -= mod_moves
            mod_moves -= 1
            rev_idx -= mod_moves
            
            rev_moves = np.zeros((M, pos_moves))
            rev_moves[np.arange(M), rev_idx] = 1
            
            for m in reversed(range(M)):
                self.move(rev_moves[m, :])        
            
            assert(self.is_solved())


class Cube3x3(Cube):
    
    #dominant orientations: Top(white), Bottom(yellow)
    #secondary: Left(blue), Right(green)

    #unique cube positions
    cublets = {"WB": 0, "WG": 1, 
               "WO": 2, "WR": 3,
               "YB": 4, "YG": 5,
               "YO": 6, "YR": 7, 
               "GO": 8, "GR": 9, 
               "BO": 10, "BR": 11,
               "WOB": 12, "WBR": 13, "WGO": 14, "WRG": 15,
               "YBO": 16, "YBR": 17, "YRG": 18, "YOG": 19, }
    inv_cublets = {v: k for k, v in cublets.items()}

    #all states for edges
    edge_pos = {"WB": 0, "BW": 1, "WG": 2, "GW": 3,
                "WO": 4, "OW": 5, "WR": 6, "RW": 7,
                "YB": 8, "BY": 9, "YG": 10, "GY": 11, 
                "YO": 12, "OY": 13, "YR": 14,"RY": 15, 
                "GO": 16, "OG": 17, "GR": 18, "RG": 19, 
                "BO": 20, "OB": 21, "BR": 22, "RB": 23, }
    inv_edge_pos = {v: k for k, v in edge_pos.items()}
           
    #all states for corners
    corner_pos = {"WOB": 0, "BWO": 1, "OBW": 2,
                  "WBR": 3, "RWB": 4, "BRW": 5,
                  "WGO": 6, "OWG": 7, "GOW": 8,
                  "WRG": 9, "GWR": 10, "RGW": 11,
                  "YBO": 12, "OYB": 13, "BOY": 14,
                  "YBR": 15, "RYB": 16, "BRY": 17,
                  "YRG": 18, "GYR": 19, "RGY": 20,
                  "YOG": 21, "GYO": 22, "OGY": 23, }
    inv_corner_pos = {v: k for k, v in corner_pos.items()}

    def __init__(self, scramble_distance=0, cube = None, history = None):
        Cube.__init__(self, scramble_distance, cube, history)

    #Perform rotation of given face 90 degrees clockwise
    #Note a counterclockwise move is 3 clockwise rotations 
    def rotate(self, move):
        self.scramble_distance+=1
        self.history.append(move)
        
        iters = 1 if move % 2 == 0 else 3

        if move in [0, 1]: # Top (white) (right cw)
            for i in range(iters):
                self.swap("WR", "WB") 
                self.swap("WB", "WO") 
                self.swap("WO", "WG") 
                
                self.swap("WBR", "WOB")
                self.swap("WOB", "WGO")
                self.swap("WGO", "WRG")
        
        elif move in [2, 3]: # Bottom (yellow) (right cw)
            for i in range(iters):
                self.swap("YR", "YB") 
                self.swap("YB", "YO") 
                self.swap("YO", "YG") 
            
                self.swap("YBR", "YBO")
                self.swap("YBO", "YOG")
                self.swap("YOG", "YRG")
                
        elif move in [4, 5]: # Left (blue) (front)
            for i in range(iters):
                self.swap("BR", "YB") 
                self.swap("YB", "BO") 
                self.swap("BO", "WB") 

                self.reorient_corner("WOB", -1)
                self.reorient_corner("WBR", 1)
                self.reorient_corner("YBO", 1)
                self.reorient_corner("YBR", -1)  ##
                self.swap("WBR", "YBR")
                self.swap("YBR", "YBO")
                self.swap("YBO", "WOB")
        
        elif move in [6, 7]: # Right (green) (back )
            for i in range(iters):
                self.swap("GR", "WG") 
                self.swap("WG", "GO") 
                self.swap("GO", "YG") 
                
                self.reorient_corner("WRG", -1)
                self.reorient_corner("WGO", 1)
                self.reorient_corner("YOG", -1)
                self.reorient_corner("YRG", 1)  ##
                self.swap("WGO", "YOG")
                self.swap("YOG", "YRG")
                self.swap("YRG", "WRG")
        
        elif move in [8, 9]: # Front (orange) (flip edge) (right cw)
            for i in range(iters):
                self.swap_flip_edge("WO", "BO")
                self.swap_flip_edge("BO", "YO")
                self.swap_flip_edge("YO", "GO")
                
                self.reorient_corner("WOB", 1)
                self.reorient_corner("WGO", -1)
                self.reorient_corner("YOG", 1)
                self.reorient_corner("YBO", -1)
                self.swap("WOB", "YBO")
                self.swap("YBO", "YOG")
                self.swap("YOG", "WGO")
        
        elif move in [10, 11]: # Back (red) (flip edge) (left ccw)
            for i in range(iters):
                self.swap_flip_edge("YR", "BR")
                self.swap_flip_edge("BR", "WR")
                self.swap_flip_edge("WR", "GR")
                
                self.reorient_corner("WBR", -1)
                self.reorient_corner("WRG", 1)
                self.reorient_corner("YBR", -1)
                self.reorient_corner("YRG", 1)
                self.swap("WRG", "YRG")
                self.swap("YRG", "YBR")
                self.swap("YBR", "WBR")
        
        else:
            print("Unknown move")
            sys.exit(1)

    #Solved state, each orientation is in default position
    def solved_cube(self):
        solved = np.zeros((20, 24), dtype = np.int32)

        solved[self.cublets["WB"], self.edge_pos["WB"]] = 1  
        solved[self.cublets["WG"], self.edge_pos["WG"]] = 1
        solved[self.cublets["WO"], self.edge_pos["WO"]] = 1
        solved[self.cublets["WR"], self.edge_pos["WR"]] = 1 
        solved[self.cublets["YB"], self.edge_pos["YB"]] = 1 
        solved[self.cublets["YG"], self.edge_pos["YG"]] = 1 
        solved[self.cublets["YO"], self.edge_pos["YO"]] = 1 
        solved[self.cublets["YR"], self.edge_pos["YR"]] = 1 
        solved[self.cublets["GO"], self.edge_pos["GO"]] = 1
        solved[self.cublets["GR"], self.edge_pos["GR"]] = 1
        solved[self.cublets["BO"], self.edge_pos["BO"]] = 1 
        solved[self.cublets["BR"], self.edge_pos["BR"]] = 1 
              
        solved[self.cublets["WOB"], self.corner_pos["WOB"]] = 1 
        solved[self.cublets["WBR"], self.corner_pos["WBR"]] = 1 
        solved[self.cublets["WGO"], self.corner_pos["WGO"]] = 1 
        solved[self.cublets["WRG"], self.corner_pos["WRG"]] = 1 
        solved[self.cublets["YBO"], self.corner_pos["YBO"]] = 1 
        solved[self.cublets["YBR"], self.corner_pos["YBR"]] = 1 
        solved[self.cublets["YRG"], self.corner_pos["YRG"]] = 1 
        solved[self.cublets["YOG"], self.corner_pos["YOG"]] = 1 
              
        return solved

class Cube2x2(Cube):
    
    #dominant orientations: Top(white), Bottom(yellow)
    #secondary: Left(blue), Right(green)

    #unique cube positions
    cublets = {"WOB": 0, "WBR": 1, "WGO": 2, "WRG": 3,
               "YBO": 4, "YBR": 5, "YRG": 6, "YOG": 7, }
    inv_cublets = {v: k for k, v in cublets.items()}
           
    #all states for corners
    corner_pos = {"WOB": 0, "BWO": 1, "OBW": 2,
                  "WBR": 3, "RWB": 4, "BRW": 5,
                  "WGO": 6, "OWG": 7, "GOW": 8,
                  "WRG": 9, "GWR": 10, "RGW": 11,
                  "YBO": 12, "OYB": 13, "BOY": 14,
                  "YBR": 15, "RYB": 16, "BRY": 17,
                  "YRG": 18, "GYR": 19, "RGY": 20,
                  "YOG": 21, "GYO": 22, "OGY": 23, }
    inv_corner_pos = {v: k for k, v in corner_pos.items()}

    def __init__(self, scramble_distance=0, cube = None, history = None):
        Cube.__init__(self, scramble_distance, cube, history)

    #Perform rotation of given face 90 degrees clockwise
    #Note a counterclockwise move is 3 clockwise rotations 
    def rotate(self, move):
        self.scramble_distance+=1
        self.history.append(move)
        
        iters = 1 if move % 2 == 0 else 3

        if move in [0, 1]: # Top (white) (right cw)
            for i in range(iters):               
                self.swap("WBR", "WOB")
                self.swap("WOB", "WGO")
                self.swap("WGO", "WRG")
        
        elif move in [2, 3]: # Bottom (yellow) (right cw)
            for i in range(iters):
                self.swap("YBR", "YBO")
                self.swap("YBO", "YOG")
                self.swap("YOG", "YRG")
                
        elif move in [4, 5]: # Left (blue) (front)
            for i in range(iters):
                self.reorient_corner("WOB", -1)
                self.reorient_corner("WBR", 1)
                self.reorient_corner("YBO", 1)
                self.reorient_corner("YBR", -1)  ##
                self.swap("WBR", "YBR")
                self.swap("YBR", "YBO")
                self.swap("YBO", "WOB")
        
        elif move in [6, 7]: # Right (green) (back )
            for i in range(iters):
                self.reorient_corner("WRG", -1)
                self.reorient_corner("WGO", 1)
                self.reorient_corner("YOG", -1)
                self.reorient_corner("YRG", 1)  ##
                self.swap("WGO", "YOG")
                self.swap("YOG", "YRG")
                self.swap("YRG", "WRG")
        
        elif move in [8, 9]: # Front (orange) (flip edge) (right cw)
            for i in range(iters):
                self.reorient_corner("WOB", 1)
                self.reorient_corner("WGO", -1)
                self.reorient_corner("YOG", 1)
                self.reorient_corner("YBO", -1)
                self.swap("WOB", "YBO")
                self.swap("YBO", "YOG")
                self.swap("YOG", "WGO")
        
        elif move in [10, 11]: # Back (red) (flip edge) (left ccw)
            for i in range(iters):
                self.reorient_corner("WBR", -1)
                self.reorient_corner("WRG", 1)
                self.reorient_corner("YBR", -1)
                self.reorient_corner("YRG", 1)
                self.swap("WRG", "YRG")
                self.swap("YRG", "YBR")
                self.swap("YBR", "WBR")
        
        else:
            print("Unknown move")
            sys.exit(1)

    #Solved state, each orientation is in default position
    def solved_cube(self):
        solved = np.zeros((8, 24), dtype = np.int32)
        solved[self.cublets["WOB"], self.corner_pos["WOB"]] = 1 
        solved[self.cublets["WBR"], self.corner_pos["WBR"]] = 1 
        solved[self.cublets["WGO"], self.corner_pos["WGO"]] = 1 
        solved[self.cublets["WRG"], self.corner_pos["WRG"]] = 1 
        solved[self.cublets["YBO"], self.corner_pos["YBO"]] = 1 
        solved[self.cublets["YBR"], self.corner_pos["YBR"]] = 1 
        solved[self.cublets["YRG"], self.corner_pos["YRG"]] = 1 
        solved[self.cublets["YOG"], self.corner_pos["YOG"]] = 1 
              
        return solved


if __name__ == '__main__':
    cube2 = Cube2x2()
    cube2.test()

    cube3 = Cube3x3()
    cube3.test()

    print("All tests pass")
