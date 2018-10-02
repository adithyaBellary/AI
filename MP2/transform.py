
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    dimension = np.empty(2)
    offsets = []
    armLinkList = arm.getArmLimit()
    start_position = arm.getArmPos()
    for limits in range(len(armLinkList)):
        minAngle, maxAngle = armLinkList[limits]
        # calculate rows/cols
        dimension[limits] = ((maxAngle - minAngle) / (granularity)) + 1
        offsets.append(minAngle)
    #initialize 2d array (maze) with dimensions
    new_maze = np.zeros((int(dimension[0]), int(dimension[1])))

    m = []
    gran = 2
    #Get wall and goal positions    
    for a in range(new_maze.shape[0]):
        t = []
        for b in range(new_maze.shape[1]):
            #check if we are at the initialization position
            
            flag = False
            angle = idxToAngle((a,b), offsets,gran )
            arm.setArmAngle(angle)
            position = arm.getArmPos()
            #set start point
            if position == start_position:
                t.append('P')
                flag = True
            #instantiate a new arm for each alpha and beta values
            arm_positions = arm.getArmPos()
            #check if we are at the goal

            if doesArmTouchGoals(arm_positions[1][1], goals):
                t.append('.')
                flag = True
            
            #check if we are at an obstacle
            if doesArmTouchObstacles(arm_positions, obstacles) or not  isArmWithinWindow(arm_positions, window):
                t.append('%')
                flag = True
            
            if flag == False:
                t.append(' ')
        m.append(t)
            
    maze = Maze(m,offsets , granularity)

    return maze