import numpy as np
import math
import itertools
from collections import deque
import heapq
import copy
import sys

import time

solution = []
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
	s = time.time()

	dim0 = len(constraints[0])
	dim1 = len(constraints[1])

	rowList = constraints[0]
	colList = constraints[1]

	# for r in rowList:
	# 	print(r)

	# print('\n')

	# for c in colList:
	# 	print(c)

	#check get domains
	# for x in rowList:
	# 	domain = getDomain()

	xDomains = [[] for i in range(dim0)]
	yDomains = [[] for i in range(dim1)]


	xWeightsUnsorted, yWeightsUnsorted = calcWeights(constraints)

	# print("xWeightsUnsorted: ", xWeightsUnsorted)
	# print("yWeightsUnsorted: ", yWeightsUnsorted)

	xWeights_indices = np.argsort(xWeightsUnsorted[:,0])[::-1]
	yWeights_indices = np.argsort(yWeightsUnsorted[:,0])[::-1]

	# print("xWeights_indices: ", xWeights_indices)
	# print("yWeights_indices: ", yWeights_indices)

	xWeights = xWeightsUnsorted[xWeights_indices]
	yWeights = yWeightsUnsorted[yWeights_indices]

	# print("xWeights: ", xWeights)
	# print("yWeights: ", yWeights)


	levelSolution = []
	# levelSolutionidx = 1
	
	temp_grid = np.zeros((dim0, dim1), dtype='int32')

	levelSolution.append(temp_grid)

	# #Tuple (Axis, index of row/col)
	# seenStack = deque([])
	# exploredStack = deque([])

	sum_x_weights = np.sum(xWeights[:,0])
	sum_y_weights = np.sum(yWeights[:,0])

	axis = ''
	changeFlag = True
	#Choose what our variable is

	# if(sum_x_weights <= sum_y_weights):
	# 	domain = getDomain(constraints[1][yWeights_indices[0]], dim0)
	# 	yDomains[yWeights_indices[0]] = domain
	# 	axis = 'y'
	# else:
	# 	domain = getDomain(constraints[0][xWeights_indices[0]], dim1)
	# 	xDomains[xWeights_indices[0]] = domain
	# 	axis = 'x'
	# xTempDomains = copy.deepcopy(xDomains)
	# yTempDomains = copy.deepcopy(yDomains)

	# print('axis: ', axis)
	global solution
	solution = initialMazeReduce(constraints)

	rows, cols = findVariableConstraintsInitial(constraints)

	# xdom = getDomain(constraints[0][0], dim1)
	# ydom = getDomain(constraints[1][0], dim0)

	# print(len(xdom[0]))
	# print(len(ydom[0]))

	# print('rows:', type(list(rows[0][0])))
	# print('cols: ', len(cols[0][0]))

	# print(rows[0])

	for r in range(len(rows)):
		# if rows[r] == []:
		# 	print('hi')
		for i in range(len(rows[r])):
			# if rows[r][i] == []:
			# 	print('hi')
			if type(rows[r][i]) is not int:
				rows[r][i] = list(rows[r][i])
				rows[r][i] = [int(j) for j in rows[r][i]]
	
	for c in range(len(cols)):
		for i in range(len(cols[c])):
			cols[c][i] = list(cols[c][i])
			cols[c][i] = [int(j) for j in cols[c][i]]
	

	xDomains = rows
	yDomains = cols

	axis = 'x'


	if(axis == 'x'): #choosing rows as variables
		#Generate all X configs
		endCase = True
		while  endCase:

			xDomains = AC3_X(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)
			yDomains = AC3_Y(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)

			# print('new x domains: \n')
			# for i in xDomains:
			# 	print(len(i))
			# print('new y domains: \n')
			# for i in yDomains:
			# 	print(len(i))

			if (sum([len(i) for i in xDomains]) == dim0) and (sum([len(i) for i in yDomains]) == dim1):
				endCase = False
			# if ((sum([len(i) for i in xDomains]) <= (dim0+dim1)) and (sum([len(i) for i in yDomains]) <= (dim0+dim1))):
			# 	endCase = False
	elif(axis == 'y'): #choosing rows as variables
		endCase = True
		while  endCase:

			yDomains = AC3_Y(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)
			xDomains = AC3_X(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)

			# print('new x domains: \n')
			# for i in xDomains:
			# 	print(len(i))
			# print('new y domains: \n')
			# for i in yDomains:
			# 	print(len(i))

			if (sum([len(i) for i in xDomains]) == dim0) and (sum([len(i) for i in yDomains]) == dim1):
				endCase = False
			# if ((sum([len(i) for i in xDomains]) <= (dim0+dim1)) and (sum([len(i) for i in yDomains]) <= (dim0+dim1))):
			# 	endCase = False

	# for row in xDomains:
	# 	for config in xDomains[row]:
	# 		if(backtrack(config, yDomains[row], 0)):
	# 			#add this config at this row to our solution
	# 			temp_grid = addConfigToState(temp_grid, 'x', row, config)
	# 		else:
	# 			pass #remove config or just try next one

	# 	#check valid solution?


	for i in range(dim0):
		x = xDomains[i][0]
		for j in range(dim1):
			temp_grid[i][j] = x[j]
	end = time.time()
	print("TIME: ", end-s)
	return temp_grid



#############################################
# 				HELPER FUNCTIONS
#############################################

def AC3_X(xDomains, xWeights_indices,yDomains, yWeights_indices, constraints, dim0, dim1):
	# print('in AC3_X')

	for x in xWeights_indices:
			#Calculate D(X) only once per row
			# print('x index: ', x)
			if(xDomains[x] == []):
				xDomains[x] = getDomain(constraints[0][x], dim1)
			# print('got the domains')
			for config_X in xDomains[x]:
				for y in yWeights_indices:
					if(yDomains[y] == []): #Only generate Y domains if we haven't stored them yet
						yDomains[y] = getDomain(constraints[1][y], dim0)
					changeFlag = True
				
					for config_Y in yDomains[y]:
						if(config_X[y] == config_Y[x]):
							changeFlag = False
					if(changeFlag == True):   	#Remove configurations that don't work
						xDomains[x].remove(config_X)	#Not sure if this line works
						break
	return xDomains

def AC3_Y(xDomains, xWeights_indices,yDomains, yWeights_indices, constraints, dim0, dim1):
	# print('in AC3_Y')
	for y in yWeights_indices:
			# print('y index: ', y)

			#Calculate D(X) only once per row
			if(yDomains[y] == []):
				yDomains[y] = getDomain(constraints[1][y], dim0)
			# print('constraints: ', constraints[1][y])
			# print('got the domains')

			#Calculate D(Y)
			for config_Y in yDomains[y]:
				# print('config y: ', config_Y)
				for x in xWeights_indices:
					# print('x: ', x)
					if(xDomains[x] == []): #Only generate Y domains if we haven't stored them yet
						xDomains[x] = getDomain(constraints[0][x], dim1)
					# print('xdomains: ',xDomains[x])
					# print('got the x domains')
					changeFlag = True
				
					for config_X in xDomains[x]:
						if(config_Y[x] == config_X[y]):
							changeFlag = False
					if(changeFlag == True):   	#Remove configurations that don't work
						yDomains[y].remove(config_Y)	#Not sure if this line works
						break
	return yDomains


# def backtrack(configuration, domains, constraints, idx):
# 	if(isValidSolution(constraints, temp_grid, axis)): #NEED to change config allowed check to compare config against all configs of opposite axis
# 		if(backtrack(configuration, domains[idx+1], constraints, levelSolution, idx)):
# 			return True
# 		else:
# 			return False
# 	return False


	# dfs_stack = deque([])	
	
	# for config in xDomains[0]:
	# 	dfs_stack.append((config, 'x', xDomains[0].index(config)))

	# while dfs_stack:
	# 	print(levelSolution[-1])
	# 	config, axis, index = dfs_stack.pop()

	# 	if(isValidSolution(constraints, levelSolution[-1])):	#check entire nonogram to see if done
	# 		break
	# 	else:
	# 		print("set backtrack")
	# 		backtrack = True

	# 	if(isConfigAllowed(xDomains, yDomains, axis, levelSolution[-1], index)):
	# 		new_temp_grid = addConfigToState(levelSolution[-1], axis, index, config)
	# 		levelSolution.append(new_temp_grid)
	# 		if(axis == 'x'):
	# 			for i in range(len(config)):
	# 				for next_config in yDomains[i]:
	# 					dfs_stack.append((next_config, 'y', yDomains[i].index(next_config)))
	# 		if(axis == 'y'):
	# 			for i in range(len(config)):
	# 				for next_config in xDomains[i]:
	# 					dfs_stack.append((next_config, 'x', xDomains[i].index(next_config)))

	# 	if(backtrack):
	# 		print("backtracking")
	# 		backtrack_config, backtrack_axis, backtrack_index = dfs_stack.pop()
	# 		while(backtrack_axis != axis):
	# 			print(dfs_stack)
	# 			backtrack_config, backtrack_axis, backtrack_index = dfs_stack.pop()
	# 		if(len(levelSolution) > 1):
	# 			levelSolution.pop()
	# 		backtrack = False

def calcWeights(constraints):
	xWeights = np.zeros((len(constraints[0]),2), dtype='int32')
	yWeights = np.zeros((len(constraints[1]),2), dtype='int32')
	for i in range(len(constraints[0])):
		for k in range(len(constraints[0][i])):
			xWeights[i][0] += constraints[0][i][k][0]
		xWeights[i][0] += (len(constraints[0][i]) -1)
		xWeights[i][1] = i
	for j in range(len(constraints[1])):
		for l in range(len(constraints[1][j])):
			yWeights[j][0] += constraints[1][j][l][0]
		yWeights[j][0] += (len(constraints[1][j]) -1)
		yWeights[j][1] = j

	return xWeights, yWeights

def addConfigToState(currentState, axis, current_index, configuration):
	# currentState: the grid/nonogram
	# configuration: what to add
	#print(axis, current_index)
	# print("Current:", current_index)
	# print("currentState.shape[0]", currentState.shape[0])
	# print("currentState.shape[1]", currentState.shape[1])
	if (axis == 'y'):
		for j in range(currentState.shape[0]):
			currentState[j][current_index] = currentState[j][current_index] | configuration[j]
	if (axis == 'x'):
		for i in range(currentState.shape[1]):
			currentState[current_index][i] = currentState[current_index][i] | configuration[i]
	return currentState

# def isValidSolution(constraints, solution, axis):
# 	solution = np.array(solution)
# 	dim0 = len(constraints[0])
# 	dim1 = len(constraints[1])
# 	if (solution.shape != (dim0, dim1)):
# 		print("Wrong dimensions, try flipping")
# 		return False
# 	if(axis == 'x'):
# 		for i in range(dim0):
# 			rowconstraints = constraints[0][i]
# 			rowcol = []
# 			for j in range(dim1):
# 				rowcol.append(solution[i][j])
# 			if not (runs(rowcol) == rowconstraints):
# 				return False
#     elif(axis == 'y'):    
# 		for j in range(dim1):
# 			colconstraints = constraints[1][j]
# 			rowcol = []
# 			for i in range(dim0):
# 				rowcol.append(solution[i][j])
# 			if not (runs(rowcol) == colconstraints):
# 				return False
# 	return True
	
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

def isConfigAllowed(xDomains, yDomains, axis, tempSolution, current_index):
	if (axis == 'x'):
		col = tempSolution[current_index].tolist()
		if (isStateAllowedHelper(xDomains[current_index], col) == False):
			return False
	if (axis == 'y'):
		# print("current_index", current_index)
		# print("size yDomains", len(yDomains))
		# print("size tempSolution", len(tempSolution[0]))
		if (isStateAllowedHelper(yDomains[current_index], tempSolution[:,current_index]) == False):
			return False
	return True

def isStateAllowedHelper(constraints, configuration):
	for con in constraints:
		inv_con = [1 if i == 0 else 0 for i in configuration]
		temp = [i | j for i, j in zip(inv_con, con)]
		if 0 not in temp:
			return True	
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

def initialMazeReduce(constraints):
#Iterate through the constraints for each column.
#We first need to minimze the size of the problem by finding
#overlap within the potential positions that must be filled
	dim0 = len(constraints[0])
	dim1 = len(constraints[1])
	solution = np.zeros((dim0, dim1))
	for i in range(0, dim0):
		numBlocks = len(constraints[0][i])
		spaceAvailable = np.zeros((1, numBlocks))
		subblockOpen = np.zeros((1, numBlocks))
		leftMostSpace = np.zeros((1, numBlocks))
		for j in range (0, numBlocks):
			spaceAvailable[0][j] = dim1-numBlocks+1
			for k in range(0, numBlocks):
				if(k!=j):
					spaceAvailable[0][j]-=constraints[0][i][k][0]
			subblockOpen[0][j] = 2 * constraints[0][i][j][0] - spaceAvailable[0][j]
			if(j>0):
				leftMostSpace[0][j] = leftMostSpace[0][j-1] + constraints[0][i][j-1][0]+1
			else:
				leftMostSpace[0][j] = 0
			if(subblockOpen[0][j] > 0):
				#Still need to add the leftmost position each can occupy
				startingPosition = leftMostSpace[0][j] + spaceAvailable[0][j] - constraints[0][i][j][0]
				for k in range(int(startingPosition), int(startingPosition+subblockOpen[0][j])):
					solution[i][k] = 1
	#Going by Rows
	for i in range(0, dim1):
		numBlocks = len(constraints[1][i])
		spaceAvailable = np.zeros((numBlocks, 1))
		subblockOpen = np.zeros((numBlocks, 1))
		leftMostSpace = np.zeros((numBlocks, 1))
		for j in range (0, numBlocks):
			spaceAvailable[j][0] = dim0-numBlocks+1
			for k in range(0, numBlocks):
				if(k!=j):
					spaceAvailable[j][0]-=constraints[1][i][k][0]
			subblockOpen[j][0] = 2 * constraints[1][i][j][0] - spaceAvailable[j][0]
			if(j>0):
				leftMostSpace[j][0] = leftMostSpace[j-1][0] + constraints[1][i][j-1][0]+1
			else:
				leftMostSpace[j][0] = 0
			if(subblockOpen[j][0] > 0):
				#Still need to add the leftmost position each can occupy
				startingPosition = leftMostSpace[j][0] + spaceAvailable[j][0] - constraints[1][i][j][0]
				for k in range(int(startingPosition), int(startingPosition+subblockOpen[j][0])):
					solution[k][i] = 1
	return solution


def findVariableConstraintsInitial(constraints):
	dim0 = len(constraints[0])
	dim1 = len(constraints[1])
	colSpace = [[] for i in range (dim1)]
	rowSpace = [[] for i in range (dim0)]
	#Iterate through each row to calculate all possible combinations for each
	for row in range(0,dim0):
		#find seed pattern
		if constraints[0][row] == []:
			rowSpace[row] = [np.array([0]*dim1)]
			print('my rowspace: ', [np.array([0]*dim1)])
			continue
		pattern = []
		currPosition = 0
		for block in range (0, len(constraints[0][row])):
			pattern.append(list(range(currPosition, constraints[0][row][block][0]+currPosition)))
			currPosition+=constraints[0][row][block][0]+1
		# print(constraints[0][0])
		# print("here")
		rowSpace[row] =variableValuesHelperRow(pattern, pattern[len(pattern)-1][0], dim1-len(pattern[len(pattern)-1])+1, len(constraints[0][row]) -1 ,dim1, row)
		print('rowspace:', rowSpace[row])
	#Iterate through each column

	for col in range(0, dim1):
		if constraints[1][col] == []:
			rowSpace[col] = [0] * dim0
			continue
		pattern = []
		currPosition = 0
		for block in range(0, len(constraints[1][col])):
			pattern.append(list(range(currPosition, constraints[1][col][block][0]+currPosition)))
			currPosition+=constraints[1][col][block][0]+1
		colSpace[col] = variableValuesHelperCol(pattern, pattern[len(pattern)-1][0], dim0-len(pattern[len(pattern)-1])+1, len(constraints[1][col])-1, dim0, col)
	return rowSpace, colSpace

def variableValuesHelperRow(pattern, first, last, currentIdx, length, currentRow):
	#record the current configuration before shifting / recursive calls
	solutionReturn = []
	#Shift current index to the right while we can
	#Recursively call for each value in the iteration
	tempPattern = copy.deepcopy(pattern)
	for i in range (first, last):
		#Take picture on lowest index
		if(currentIdx == 0):
			newArray = np.zeros(length)
			for k in range(0, len(tempPattern)):
				for j in range(0, len(tempPattern[k])):
					newArray[tempPattern[k][j]] =1
			comparison = solution[currentRow] - newArray
			#CHANGE THIS CHANGE THIS
			if(all(k<=0 for k in comparison)):
				solutionReturn.append(newArray)
		#Shifting each value in the current index, i.e. moving every block in block list over 1
		if(currentIdx>0):
			arrayReturned = variableValuesHelperRow(tempPattern, tempPattern[currentIdx-1][0], i-len(tempPattern[currentIdx-1]), currentIdx-1, length, currentRow)
			for k in range (0, len(arrayReturned)):
				solutionReturn.append(arrayReturned[k])
		for j in range (0, len(pattern[currentIdx])):
			tempPattern[currentIdx][j]+=1
		# We go one layer deeper with the shift
	return solutionReturn


def variableValuesHelperCol(pattern, first, last, currentIdx, length, currentCol):
	#record the current configuration before shifting / recursive calls
	solutionReturn = []
	#Shift current index to the right while we can
	#Recursively call for each value in the iteration
	tempPattern = copy.deepcopy(pattern)
	for i in range (first, last):
		#Take picture on lowest index
		if(currentIdx == 0):
			newArray = np.zeros(length)
			for k in range(0, len(tempPattern)):
				for j in range(0, len(tempPattern[k])):
					newArray[tempPattern[k][j]] =1
			solutionCheck = np.zeros(length)
			for m in range(0, length):
				solutionCheck[m] = solution[m][currentCol]
			comparison = solutionCheck - newArray
			#CHANGE THIS CHANGE THIS
			if(all(k<=0 for k in comparison)):
				solutionReturn.append(newArray)
		#Shifting each value in the current index, i.e. moving every block in block list over 1
		if(currentIdx>0):
			arrayReturned = variableValuesHelperCol(tempPattern, tempPattern[currentIdx-1][0], i-len(tempPattern[currentIdx-1]), currentIdx-1, length, currentCol)
			for k in range (0, len(arrayReturned)):
				solutionReturn.append(arrayReturned[k])
		for j in range (0, len(pattern[currentIdx])):
			tempPattern[currentIdx][j]+=1
		# We go one layer deeper with the shift
	return solutionReturn


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

def view(solution):
	height = solution.shape[0]
	width = solution.shape[1]
	block_size = 20
	offset = 32
	
	display = pygame.display.set_mode(
		(
			width * block_size + 2 * offset,
			height * block_size + 2 * offset, 
			
		), 0, 0)
	
	max_color = float(max(1, np.max(solution)))
	for row in range(len(solution)):
		for col in range(len(solution[0])):
			shade = 255 - (solution[row,col]/max_color)*255
			color = (shade, shade, shade)
			rect = pygame.Rect(offset + col * block_size, 
							   offset + row * block_size, 
							   block_size, 
							   block_size
					)
			pygame.draw.rect(display, color, rect)
	pygame.display.update()
	
	time.sleep(25)
