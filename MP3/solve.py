import numpy as np
import math
import itertools
from collections import deque
import heapq
import copy
import sys

import time

def solve(constraints):
	"""
	Implement me!!!!!!!
	This function takes in a set of constraints. The first dimension is the axis
	to which the constraints refer to. The second dimension is the list of constraints
	for some axis index pair. The third demsion is a single constraint of the form 
	[i,j] which means a run of i js. For example, [4,1] would correspond to a block
	[1,1,1,1].
	
	The return value of this function should be a numpy array that satisfies all
	of the constraints.


	A puzzle will have the constraints of the following format:
	
	
	array([
		[list([[4, 1]]), 
		 list([[1, 1], [1, 1], [1, 1]]),
		 list([[3, 1], [1, 1]]), 
		 list([[2, 1]]), 
		 list([[1, 1], [1, 1]])],
		[list([[2, 1]]), 
		 list([[1, 1], [1, 1]]), 
		 list([[3, 1], [1, 1]]),
		 list([[1, 1], [1, 1]]), 
		 list([[5, 1]])]
		], dtype=object)
	
	And a corresponding solution may be:

	array([[0, 1, 1, 1, 1],
		   [1, 0, 1, 0, 1],
		   [1, 1, 1, 0, 1],
		   [0, 0, 0, 1, 1],
		   [0, 0, 1, 0, 1]])



	Consider a more complicated set of constraints for a colored nonogram.

	array([
	   [list([[1, 1], [1, 4], [1, 2], [1, 1], [1, 2], [1, 1]]),
		list([[1, 3], [1, 4], [1, 3]]), 
		list([[1, 2]]),
		list([[1, 4], [1, 1]]), 
		list([[2, 2], [2, 1], [1, 3]]),
		list([[1, 2], [1, 3], [1, 2]]), 
		list([[2, 1]])],
	   [list([[1, 3], [1, 4], [1, 2]]),
		list([[1, 1], [1, 4], [1, 2], [1, 2], [1, 1]]),
		list([[1, 4], [1, 1], [1, 2], [1, 1]]), 
		list([[1, 2], [1, 1]]),
		list([[1, 1], [2, 3]]), 
		list([[1, 2], [1, 3]]),
		list([[1, 1], [1, 1], [1, 2]])]], 
		dtype=object)

	And a corresponding solution may be:

	array([
		   [0, 1, 4, 2, 1, 2, 1],
		   [3, 4, 0, 0, 0, 3, 0],
		   [0, 2, 0, 0, 0, 0, 0],
		   [4, 0, 0, 0, 0, 0, 1],
		   [2, 2, 1, 1, 3, 0, 0],
		   [0, 0, 2, 0, 3, 0, 2],
		   [0, 1, 1, 0, 0, 0, 0]
		 ])


	"""
	# s = time.time()


	dim0 = len(constraints[0])
	dim1 = len(constraints[1])

	rowList = constraints[0]
	colList = constraints[1]

	xDomains = []
	yDomains = []

	# for i in rowList:
	# 	print(i)

	# print('rowlist: ', rowList)
	# print('colList: ', colList)

	for x in range(dim0):
		# print("rowlist: ", rowList[x])
		domain = getDomain(rowList[x], dim1)
		#heapq.heappush(xDomains, (len(domain),'x', x, domain))
		xDomains.append((len(domain),'x', x, domain))
	
	for y in range(dim1):
		# print('colList: ', colList[y])
		domain = getDomain(colList[y], dim0)
		#heapq.heappush(yDomains, (len(domain),'y', y, domain))
		yDomains.append((len(domain),'y', y, domain))

	# for i in xDomains:
	# 	print(i, '\n')
	#sort the x and y domains so that we can start with the most constrained index
	# heapq.heapify(xDomains)
	# heapq.heapify(yDomains)
	
	# print('xdomains')
	# for i in xDomains:
	# 	print(i, '\n')

	# print('ydomains')
	# for i in yDomains:
	# 	print(i, '\n')

	
	# if xDomains[0][0] < yDomains[0][0]:
	# 	min_constrained = xDomains[0]
	# else:
	# 	min_constrained = yDomains[0]

	

	levelSolution = []
	levelSolutionidx = 1
	
	temp_grid = np.zeros((dim0, dim1))
	temp_grid = temp_grid.astype(int)

	levelSolution.append(temp_grid)

	# x or y, index, one possible configuration
	curr_x_idx = 0     #X indices that we've seen
	curr_y_idx = 0	   #Y indices that we've seen

	x_idx = []
	y_idx = []

	dfs_stack = deque( [ (xDomains[0][1], xDomains[0][2], xDomains[0][3][0], levelSolutionidx) ] )
	#how to add onto stack
	for config in xDomains[0][3]:
		dfs_stack.append((xDomains[0][1], xDomains[0][2], config, levelSolutionidx))

	while dfs_stack:

		axis, current_index, current_config, gar = dfs_stack.pop()
		print("current tuple:", axis, current_index, current_config, levelSolutionidx)
		text = input("prompt:")
		if (1==0): #isNonogramAllowed(xDomains, yDomains, levelSolution[levelSolutionidx-1]) == True):
			break

		else:
			# check is state allowed
			print("before isstate allowed, current config", current_config)
			#print("constraints", constraints)
			new_temp_grid = copy.deepcopy(levelSolution[levelSolutionidx-1]) #set new tempgrid equal to old good temp grid, then add new config in to ee if it works
			new_temp_grid = addConfigToState(new_temp_grid, axis, current_index, current_config)
			print("constraints: ", constraints)			
			print(new_temp_grid)
			text = input("prompt:")
			if (axis == 'x'):
				#if(isSingleStateAllowed(xDomains[curr_x_idx][3], current_config) == True):
				tempAllowed = isStateAllowed(xDomains, yDomains, 'x', new_temp_grid) #ISSUE: comparing wrong. comparing an x and y column. will not fit
			if (axis == 'y'):
				#if(isSingleStateAllowed(yDomains[curr_y_idx][3], current_config) == True):
				tempAllowed = isStateAllowed(xDomains, yDomains, 'y', new_temp_grid) #check if this added state is allowed 
			# if allowed, add new configs at unseen opposite coordinate
			if (tempAllowed == True):
				print("temp is allowed")
				text = input("prompt:")
				# Add allowed tuple onto the temp grid
				# new_temp_grid = copy.deepcopy(levelSolution[levelSolutionidx-1]) #set new tempgrid equal to old good temp grid, then add new config in to ee if it works
				# new_temp_grid = addConfigToState(new_temp_grid, axis, current_index, current_config)
				print("Solution Before appending: ",levelSolution)
				#print("\n \n TEMP GRID HERE \n \n ",new_temp_grid)
				if (levelSolutionidx >= len(levelSolution)):
					levelSolution.append(new_temp_grid)
				else:
					levelSolution[levelSolutionidx] = copy.deepcopy(new_temp_grid)
					print("Solution Index: ", levelSolutionidx)
				print("Solution after appending", levelSolution)
				text = input("prompt:")
				#return levelSolution
				#check whether currently on x or y axis
				if (axis == 'x'): #xaxis
					curr_y_idx += 1
					levelSolutionidx += 1
					#print("yDomains: ",yDomains)
					print("curr_y_idx: ", curr_y_idx)
					for config in yDomains[curr_y_idx][3]: 
						dfs_stack.append((yDomains[curr_y_idx][1], yDomains[curr_y_idx][2], config, levelSolutionidx))
				if (axis == 'y'): #yaxis
					curr_x_idx += 1
					levelSolutionidx += 1
					for config in xDomains[curr_x_idx][3]: #change index smart, find index that hasnt been searched yet (next best constrained index in Domain)
						dfs_stack.append((xDomains[curr_x_idx][1], xDomains[curr_x_idx][2], config, levelSolutionidx))

			# else just set down the first configuration in the list of domains
			else:
				temp_axis, temp_current_index, temp_current_config, temp_levelSolutionidx = dfs_stack.pop()
				print("Level Soln Index: ",levelSolutionidx)
				print("Temp Level Index: ", temp_levelSolutionidx)
				if (temp_axis != axis or temp_current_index != current_index):
					if (temp_axis == 'x'):
						curr_x_idx = temp_current_index
						levelSolutionidx = temp_levelSolutionidx
					if (temp_axis == 'y'):
						curr_y_idx = temp_current_index
						levelSolutionidx = temp_levelSolutionidx
				dfs_stack.append((temp_axis, temp_current_index, temp_current_config, temp_levelSolutionidx))
	print("constraints: ", constraints)			
	print("level solution:", levelSolution)
	print("current indexes: ", curr_x_idx, curr_y_idx)
	return levelSolution[levelSolutionidx-1]

	

#############################################
# 				HELPER FUNCTIONS
#############################################

def addConfigToState(currentState, axis, current_index, configuration):
	# currentState: the grid/nonogram
	# configuration: what to add
	print(axis, current_index)
	if (axis == 'y'):
		for j in range(currentState.shape[1]):
			currentState[current_index][j] = currentState[current_index][j] | configuration[j]
	if (axis == 'x'):
		for i in range(currentState.shape[0]):
			currentState[i][current_index] = currentState[i][current_index] | configuration[i]
	return currentState

def isStateAllowed(xDomains, yDomains, axis, tempSolution):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing
	#print(constraints)
	#if (axis == 'x'):
	for i in range(len(yDomains)):
		col = tempSolution[i].tolist()
		if (isStateAllowedHelper(yDomains[i][3], col) == False):
			return False
	#if (axis == 'y'):
	for j in range(len(xDomains)):
		if (isStateAllowedHelper(xDomains[j][3], tempSolution[:,j]) == False):
			return False
	return True

def isStateAllowedHelper(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing
	for con in constraints:
		inv_con = [1 if i == 0 else 0 for i in configuration]
		temp = [i | j for i, j in zip(inv_con, con)]
		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
		if 0 not in temp:
			#there is a configuration that works
			return True	

	return False

def isSingleStateAllowed(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing
	for con in constraints:
		inv_con = [1 if i == 0 else 0 for i in configuration]
		temp = [i | j for i, j in zip(inv_con, con)]
		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
		if 0 not in temp:
			#there is a configuration that works
			return True	

	return False


def isNonogramAllowedHelper(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing
	for con in constraints:
		con = [0 if i == 0 else 1 for i in configuration]
		temp = [i ^ j for i, j in zip(con, constraints)]
		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
		if 0 not in temp:
			#there is a configuration that works
			return True	

	return False

def isNonogramAllowed(xDomains, yDomains, tempSolution):
	#constraint: list of lists
	#tempSolution: The configuration we are testing
	#print(constraints)
	print(tempSolution)
	for x in range(xDomains.shape[0]):
		col = tempSolution[x].tolist()
		if (isNonogramAllowedHelper(xDomains[x][3], col) == False):
			return False
	for y in range(yDomains.shape[1]):
		if (isNonogramAllowedHelper(yDomains[y][3], tempSolution[:,y]) == False):
			return False
	
	return True

def getDomain(constraints, dimension):

	runs = [i for i, j in constraints]
	colors  = [j for i, j in constraints]
	#calculate number of 0s we have to place
	num_zeros = dimension - sum(runs)

	q = deque([colors])
	masterList_ = []
	

	while q:
		#while the queue is empty
		arr = q.pop()
		if len(arr) == (len(colors) + num_zeros):
			if (isValid(arr)) and (arr not in masterList_):
				masterList_.append(arr)
		else:
			#insert the zeros
			for i in range(len(arr) + 1):
				temp = arr.copy()
				temp.insert(i, 0)
				if temp not in q:
					# print('temp: ', temp)
					# print(len(temp))
					q.append(temp)
	
	
	#Now incorporate runs
	newMasterList = []
	for l in masterList_:
		runCounter = 0
		newList = []
		for i in range(len(l)):
			if l[i] == 0:
				newList.append(0)
			else:
				for j in range(runs[runCounter]):
					newList.append(l[i])
				runCounter += 1
				# print(runCounter)
		newMasterList.append(newList)
	
					
	# for i in newMasterList:
	#     print(i)
	return newMasterList

def isValid(posConfig):
    #checks to see if this array is allowed
    for i in range(len(posConfig)-1):
        if (posConfig[i] == posConfig[i+1]) and (posConfig[i] != 0):
            return False
    return True
