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

##### Helper Functions ####

#Calculate the distance from a line to a point
#Attempt 1
#def distance(P1, P2, Point):
#    #all arguments are tuples for (x, y) coordinates
#    num = abs( ( P2[1] - P1[1] )*Point[0] - ( P2[0] - P1[0] )*Point[1] + P2[0]*P1[1] -  P2[1]*P1[0] )
#    den = math.sqrt( (P2[1] - P1[1])**2 + (P2[0] - P1[0])**2  )
#    return num / den

#Calculate the distance from a point to a line
#Attempt 2
def distance2(P1, P2, Point):
    #unpack the tuples to x and y corrdinates
    x1, y1 = P1
    x2, y2 = P2

    x0, y0 = Point

    #find slope
    m = (y2 - y1) / (x2 - x1)
    intercept = y1 - m*x1

    a = -1*m
    b = 1
    c = -1*intercept

    d = abs( a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)

    return d


def distance(end, Point):
    #unpack the tuples to x and y corrdinates
    x1, y1 = end

    x0, y0 = Point

    #find slope
    dist_sq = (y1 - y0)**2 + (x1 - x0)**2

    return math.sqrt(dist_sq)

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

    # print(armPos)

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

            if distance2(arm[0], arm[1], ob_coord) < rad:
                #if the arm is touching the obstacle, return True
                print("TOUCHING AN OBSTACLE")
            
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

    for arm in armEnd:
        #for each arm link - (start, end)
        for g in goals:
            #for each obstacle
            rad = g[2]
            #unpack the tuple to get the coordinates of the obstacle
            goal_coord = (g[0], g[1])
            #if the distance from the end to the goal is less than the radius return True

            if distance(arm, goal_coord) < rad:
                #if the arm is touching the obstacle, return True
                print("TOUCHING AN GOAL")
                return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """

    return True

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
            points.append(armPos[i][0])
            points.append(armPos[i][1])


    #check of any of the points hits the boundary lines
    for pt in points:
        #for each point
        for b in boundaries:
            #for each of the boundaries (lines)
            #check if the point touches the boundary
            if distance(b[0], b[1], pt) <= 10:
                print("TOUCHING A BOUNDARY LINEs")
                return False
    
    
    return True