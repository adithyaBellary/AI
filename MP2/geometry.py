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
def distance(P1, P2, Point):
   #all arguments are tuples for (x, y) coordinates
	x1, y1 = P1
	x2, y2 = P2
	x0, y0 = Point
	dx = x2-x1
	dy = y2-y1

	u =  ((x0 - x1) * dx + (y0 - y1) * dy) / float(dx**2 + dy**2)

	if u > 1:
		u = 1
	elif u < 0:
		u = 0

	x = x1 + u * dx
	y = y1 + u * dy

	dist = math.sqrt((x - x0)**2 + (y - y0)**2)

	return dist

#Calculate the distance from a point to a line
#Attempt 2
def distance2(start, end, obstacle):
	#unpack the tuples to x and y corrdinates
	x1, y1 = start
	x2, y2 = end

	x0, y0 = obstacle
	#find slope
	m = (y2 - y1) / (x2 - x1)
	intercept = y1 - m*x1

	a = 1*m
	b = -1
	c = 1*intercept

	d = abs( a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)
	return d

def point_distance(end, Point):
	#unpack the tuples to x and y corrdinates
	x1, y1 = end

	x0, y0 = Point
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
	radian = float(angle*(math.pi / 180))

	return (start[0] + length * math.cos(radian), start[1] - length * math.sin(radian))

def doesArmTouchObstacles(armPos, obstacles):
	"""Determine whether the given arm links touch obstacles

		Args:
			armPos (list): start and end position of all arm links [(start, end)]
			obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

		Return:
			True if touched. False it not.
	"""    
	for i in range(len(armPos)):
		arm = armPos[i]
		#for each arm link - (start, end)
		for j in range(len(obstacles)):
			obs = obstacles[j]
			#for each obstacle
			rad = obs[2]
			#unpack the tuple to get the coordinates of the obstacle
			ob_coord = (obs[0], obs[1])
			#if the distance from the line to the obstacle is less than the radius return True

			d = distance(arm[0], arm[1], ob_coord)
			if (arm[0][0] != arm[1][0]) and (d <= (rad)):
				#first condition to avoid divide by 0 (the line is vertical and slope is inf)
				#if the arm is touching the obstacle, return True
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
	for g in goals:
		#for each obstacle
		rad = g[2]
		#unpack the tuple to get the coordinates of the obstacle
		goal_coord = (g[0], g[1])
		#if the distance from the end to the goal is less than the radius return True
		if point_distance(armEnd, goal_coord) <= (rad+0.1):
			# print(point_distance(armEnd, goal_coord))
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

	width = window[0]
	height = window[1]

	for arm in armPos[1:]:
		#for each arm (ignore the first arm)
		start = arm[0]
		end = arm[1]

		if (start[1] > height) or (start[1] < 0):
			return False
		if (start[0] > width) or (start[0] < 0):
			return False
		if (end[1] > height) or (end[1] < 0):
			return False
		if (end[0] > width) or (end[0] < 0):
			return False
		
	return True