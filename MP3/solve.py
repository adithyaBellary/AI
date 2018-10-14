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

	xDomains = [[]]*dim1
	yDomains = [[]]*dim0

	# for x in range(dim0):
	# 	domain = getDomain(rowList[x], dim1)
	# 	#heapq.heappush(xDomains, (len(domain),'x', x, domain))
	# 	xDomains.append((len(domain),'x', x, domain))
	
	# for y in range(dim1):
	# 	domain = getDomain(colList[y], dim0)
	# 	#heapq.heappush(yDomains, (len(domain),'y', y, domain))
	# 	yDomains.append((len(domain),'y', y, domain))

	#Sorted weights for each row and column
	#Order (weight, index)
	xWeights, yWeights = calcWeights(constraints)

	levelSolution = []
	levelSolutionidx = 1
	
	temp_grid = np.zeros((dim0, dim1))
	temp_grid = temp_grid.astype(int)

	levelSolution.append(temp_grid)

	curr_x_idx = 0     #Current col we're checking
	curr_y_idx = -1	   #Current row we're checking

	#Tuple (Axis, index of row/col, level index)
	rowcol_seen = deque([])

	#Domains = (list of configurations, axis, index of row/col)
	if(xWeights[len(xWeights)-1][0] <= yWeights[len(yWeights)-1][0]):
		domain = getDomain(constraints[0][yWeights[len(yWeights)-1][1]], dim1)
		yDomains[yWeights[len(yWeights)-1][1]] = ((domain, 'y'))
		rowcol_seen.append(('y', yWeights[len(yWeights)-1][1], levelSolutionidx))
	else:
		domain = getDomain(constraints[1][xWeights[len(xWeights)-1][1]], dim0)
		xDomains[xWeights[len(xWeights)-1][1]] = ((domain, 'x'))
		rowcol_seen.append(('x', xWeights[len(xWeights)-1][1], levelSolutionidx))

	
	while rowcol_seen:

		if (isValidSolution(xDomains, yDomains, levelSolution[levelSolutionidx-1]) == True):	#CHANGE HELPER FUNCTION
			break

		else:
			axis, rowcol_index, level_index = rowcol_seen.pop()
			if(rowcol_index not in rowcol_seen)
				if(axis == 'x'):
					domain = getDomain(constraints[0][rowcol_index], dim1)
				elif(axis == 'y'):
					domain = getDomain(constraints[1][rowcol_index], dim0)
				rowcol_seen.append((axis, rowcol_index))

			new_temp_grid = copy.deepcopy(levelSolution[levelSolutionidx-1]) #set new tempgrid equal to old good temp grid, then add new config in to ee if it works
			new_temp_grid = addConfigToState(new_temp_grid, axis, current_index, current_config)

			#Pick first configuration
			#For each cell in config, add all indices of opposite axis onto queue proper axis queue
				#Remove first, use as new start
			#Continue loop

			for i in current_config:
				if(i != 0):
					if(axis == 'x'):
						x_seen.append(i)
					if(axis == 'y'):
						y_seen.append(i)


			if (axis == 'x'):
				tempAllowed = isConfigAllowed(xDomains, yDomains, 'x', new_temp_grid, current_index)
			if (axis == 'y'):
				tempAllowed = isConfigAllowed(xDomains, yDomains, 'y', new_temp_grid, current_index) 
			if (tempAllowed == True):
				if (levelSolutionidx >= len(levelSolution)):
					levelSolution.append(new_temp_grid)
				else:
					levelSolution[levelSolutionidx] = copy.deepcopy(new_temp_grid)
				if (axis == 'x'): #xaxis
					curr_y_idx += 1
					levelSolutionidx += 1
					for config in yDomains[curr_y_idx][3]: 
						dfs_stack.append((yDomains[curr_y_idx][1], yDomains[curr_y_idx][2], config, levelSolutionidx))
				if (axis == 'y'): #yaxis
					curr_x_idx += 1
					levelSolutionidx += 1
					for config in xDomains[curr_x_idx][3]: 
						dfs_stack.append((xDomains[curr_x_idx][1], xDomains[curr_x_idx][2], config, levelSolutionidx))

			# else just set down the first configuration in the list of domains
			else:
				temp_axis, temp_current_index, temp_current_config, temp_levelSolutionidx = dfs_stack.pop()
				if (temp_axis != axis or temp_current_index != current_index):
					if (temp_axis == 'x'):
						curr_x_idx = copy.copy(temp_current_index)
						curr_y_idx = copy.copy(temp_current_index) - 1
						levelSolutionidx = copy.copy(temp_levelSolutionidx)
					if (temp_axis == 'y'):
						curr_x_idx = copy.copy(temp_current_index)
						curr_y_idx = copy.copy(temp_current_index)
						levelSolutionidx = copy.copy(temp_levelSolutionidx)
				dfs_stack.append((temp_axis, temp_current_index, temp_current_config, temp_levelSolutionidx))
	print("levelSolutionidx: ", levelSolutionidx)
	print("current indexes: ", curr_x_idx, curr_y_idx)
	solutionFound = np.array(levelSolution[levelSolutionidx-1])
	print(solutionFound)
	return solutionFound

	

#############################################
# 				HELPER FUNCTIONS
#############################################

def calcWeights(constraints):
	xWeights = np.zeros((len(constraints[0]),2), dtype='int32')
	yWeights = np.zeros((len(constraints[1]),2), dtype='int32')
	for i in range(len(constraints[0])):
		for k in range(len(constraints[0][i])):
			xWeights[i][0] += constraints[0][i][k][0]
		xWeights[i][0] += (len(constraints[0][i]) -1)
		xWeights[i][1] = i
	for j in range(len(constraints[1])):
		for k in range(len(constraints[1][j])):
			yWeights[j][0] += constraints[1][j][k][0]
		yWeights[j][0] += (len(constraints[1][j]) -1)
		yWeights[j][1] = j

	# print("unsorted:")
	# print(xWeights)
	# print(yWeights)
	xWeights = xWeights[np.argsort(xWeights[:,0])]
	yWeights = yWeights[np.argsort(yWeights[:,0])]

	# print("sorted")
	# print(xWeights)
	# print(yWeights)

	return xWeights, yWeights

def addConfigToState(currentState, axis, current_index, configuration):
	# currentState: the grid/nonogram
	# configuration: what to add
	#print(axis, current_index)
	if (axis == 'y'):
		for j in range(currentState.shape[1]):
			currentState[j][current_index] = currentState[j][current_index] | configuration[j]
	if (axis == 'x'):
		for i in range(currentState.shape[0]):
			currentState[current_index][i] = currentState[current_index][i] | configuration[i]
	return currentState

def isValidSolution(constraints, solution):
    """Returns True if solution fits constraints, False otherwise"""
    solution = np.array(solution)
    dim0 = len(self.constraints[0])
    dim1 = len(self.constraints[1])
    if solution.shape != (dim0, dim1):
        return False
    for i in range(dim0):
        constraints = self.constraints[0][i]
        rowcol = []
        for j in range(dim1):
            rowcol.append(solution[i][j])
        if not (runs(rowcol) == constraints):
            return False
        
    for j in range(dim1):
        constraints = self.constraints[1][j]
        rowcol = []
        for i in range(dim0):
            rowcol.append(solution[i][j])
        if not (runs(rowcol) == constraints):
            return False
    return True
    
def runs(rowcol):
	"""
	Returns the set of nonzero runs for a given row or column
	For example, the row or column [1,1,0,2,2,0,1,2,1] returns
	[[2,1],[2,2],[1,1],[1,2],[1,1]]
	"""
	run = []
	curr_run = [1, rowcol[0]]
	rowcol.append(0)
	for i in range(1,len(rowcol)):
	    if rowcol[i] != curr_run[1]:
	        if curr_run[1] != 0:
	            run.append(curr_run)
	        curr_run = [1,rowcol[i]]
	    else: 
	        curr_run[0] += 1
	return run

# def isStateAllowed(xDomains, yDomains, axis, tempSolution, current_index):
# 	#temp_grid: our full grid so far
# 	#constraint: list of lists
# 	#configuration: The configuration we are testing
# 	#print(constraints)
# 	#if (axis == 'x'):
# 	for i in range(len(xDomains)):
# 		col = tempSolution[i].tolist()
# 		if (isStateAllowedHelper(xDomains[i][3], col) == False):
# 			return False
# 	#if (axis == 'y'):
# 	for j in range(len(yDomains)):
# 		if (isStateAllowedHelper(yDomains[j][3], tempSolution[:,j]) == False):
# 			return False
# 	return True

def isStateAllowedHelper(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing
	for con in constraints:
		inv_con = [1 if i == 0 else 0 for i in configuration]
		temp = [i | j for i, j in zip(inv_con, con)]
		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
		#print("constraint:", con)
		if 0 not in temp:
			#print("True temp state:", temp)
			#there is a configuration that works
			return True	
	#print("False temp state:", temp)
	return False

# def isNonogramAllowedHelper(constraints, configuration):
# 	#temp_grid: our full grid so far
# 	#constraint: list of lists
# 	#configuration: The configuration we are testing
# 	for con in constraints:
# 		temp = [1 if (configuration[i]==con[i]) else 0 for i in range(len(configuration))]
# 		#temp = [i ^ j for i, j in zip(fig1, con)]
# 		#print("temp sol in nonogram allowed?", temp)
# 		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
# 		if 0 not in temp:
# 			#there is a configuration that works
# 			return True	

# 	return False

# def isNonogramAllowed(xDomains, yDomains, tempSolution):
# 	#constraint: list of lists
# 	#tempSolution: The configuration we are testing
# 	#print(constraints)
# 	#print(tempSolution)
# 	for i in range(len(xDomains)):
# 		col = tempSolution[i].tolist()
# 		if (isNonogramAllowedHelper(xDomains[i][3], col) == False):
# 			return False
# 	for j in range(len(yDomains)):
# 		if (isNonogramAllowedHelper(yDomains[j][3], tempSolution[:,j]) == False):
# 			return False
	
# 	return True

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
