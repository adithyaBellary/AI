# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    ## subtract length * math.sin(angle)???
    ## top-left is (0,0)
    # angle = 2 * math.pi / 180
    return (start[0] + length * math.cos(angle), start[1] + length * math.sin(angle))

def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """    
    # x = 
    # for i in obstacles:
    #     print(i)
    # print(armPos)
    # print('\n')

    for arm in armPos:
        #for each arm link - (start, end)
        for obs in obstacles:
            #for each obstacle
            rad = obs[2]
            #unpack the tuple to get the coordinates of the obstacle
            ob_coord = (obs[0], obs[1])
            #if the distance from the line to the obstacle is less than the radius return True

            #might need to make it <= rad???
            #will probably affect how the configuration space is transformed
            if distance(arm[0], arm[1], ob_coord) < rad:
                #if the arm is touching the obstacle, return True
                print("TOUCHING")
            
                return True
    return False



def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """
    width = window[0]
    height = window[1]
    boundaries = [ ((0,0), (width, 0)), ##X-axis
                   ((0,0), (0,height)), ##Y-axis
                   ((width, 0), (width, height)), ##vertical line at (width, 0) 
                   ((0,height), (width, height)) ] ##horizontal line at (0, height)
    points = []
    for i in range(len(armPos)):
        if i == 0:
            #if we are on the first link
            #we only care about the end point, not the start because that point does not move
            points.append(armPos[i][1])
        else:
            #if not, add both the start and end points of the links
            #This should take into account if we have more than 2 links???
            points.append(armPos[i][0], armPos[i][1])


    #check of any of the links hits the boundary lines

    
    return True

    ##### Helper Functions ####

    #Calculate the distance from a line to a point
    def distance(P1, P2, Point):
        #all arguments are tuples for (x, y) coordinates
        num = abs( ( P2[1] - P1[1] )*Point[0] - ( P2[0] - P1[0] )*Point[1] + P2[0]*P1[1] -  P2[1]*P1[0] )
        den = math.sqrt( (P2[1] - P1[1])**2 + (P2[0] - P1[0])**2  )
        return num / den