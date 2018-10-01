
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

    # need angles values
    dimension = np.empty(3)

    armLinkList = getArmLimit()
    for limits in range(len(armLinkList)):
        minAngle, maxAngle = armLinkList[limits]
        # calculate rows/cols
        dimension[limits] = ((maxAngle - minAngle) / (granularity)) + 1

    #initialize 2d array (maze) with dimensions
    new_maze = np.empty((dimensions[0], dimensions[1]))
    
    # Get starting position
    alpha, beta = arm.getArmAngle()
    
    #Get wall and goal positions    
    for row in range(len(new_maze)):
        for col in range(len(new_maze[0])):
            if(obstacles.index((row,col))):
                new_maze[row][col] = '%'
            elif(goals.index((row,col))):
                new_maze[row][col] = '.'


    #maze = Maze(new_maze, , granularity)

    pass