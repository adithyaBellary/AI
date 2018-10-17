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

	global dim0
	dim0 = len(constraints[0])
	global dim1
	dim1 = len(constraints[1])

	rowList = constraints[0]
	colList = constraints[1]

	xDomains = [[] for i in range(dim0)]
	yDomains = [[] for i in range(dim1)]


	xWeightsUnsorted, yWeightsUnsorted = calcWeights(constraints)

	xWeights_indices = np.argsort(xWeightsUnsorted[:,0])[::-1]
	yWeights_indices = np.argsort(yWeightsUnsorted[:,0])[::-1]

	xWeights = xWeightsUnsorted[xWeights_indices]
	yWeights = yWeightsUnsorted[yWeights_indices]


	levelSolution = []
	temp_grid = np.zeros((dim0, dim1), dtype='int32')

	levelSolution.append(temp_grid)

	sum_x_weights = np.sum(xWeights[:,0])
	sum_y_weights = np.sum(yWeights[:,0])

	axis = ''
	changeFlag = True

	colCon = [[] for i in range (dim1)]
	rowCon = [[] for i in range (dim0)]

	global solution
	solution = Reduce(constraints)

	xDomains, yDomains = findInitialCon(constraints, rowCon, colCon)

	for r in range(len(xDomains)):
		for i in range(len(xDomains[r])):
			if type(xDomains[r][i]) is not int:
				xDomains[r][i] = list(xDomains[r][i])
				xDomains[r][i] = [int(j) for j in xDomains[r][i]]
	
	for c in range(len(yDomains)):
		for i in range(len(yDomains[c])):
			yDomains[c][i] = list(yDomains[c][i])
			yDomains[c][i] = [int(j) for j in yDomains[c][i]]
	
	axis = 'x'
	if(axis == 'x'): 
		endCase = True
		while  endCase:

			xDomains = AC3_X(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)
			yDomains = AC3_Y(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)

			if (sum([len(i) for i in xDomains]) == dim0) and (sum([len(i) for i in yDomains]) == dim1):
				endCase = False
	elif(axis == 'y'): #choosing rows as variables
		endCase = True
		while  endCase:

			yDomains = AC3_Y(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)
			xDomains = AC3_X(xDomains, xWeights_indices, yDomains, yWeights_indices, constraints, dim0, dim1)

			if (sum([len(i) for i in xDomains]) == dim0) and (sum([len(i) for i in yDomains]) == dim1):
				endCase = False

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
	if (axis == 'y'):
		for j in range(currentState.shape[0]):
			currentState[j][current_index] = currentState[j][current_index] | configuration[j]
	if (axis == 'x'):
		for i in range(currentState.shape[1]):
			currentState[current_index][i] = currentState[current_index][i] | configuration[i]
	return currentState
	
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

def Reduce(constraints):
	solution = np.zeros((dim0, dim1))
	#Going by Rows
	temp  = np.zeros(dim0)
	for i in range(dim1):
		numBlocks = len(constraints[1][i])
		spaceAvailable = np.zeros((numBlocks, 1))
		sub = np.zeros((numBlocks, 1))
		leftMostSpace = np.zeros((numBlocks, 1))
		for z in range(len(temp)):
			if (temp[z] == 1):
				temp[z] = 0
			elif (temp[z] == 0):
				temp[z] = 1
		for j in range (numBlocks):
			space = dim0-numBlocks+1
			spaceAvailable[j][0] = space
			for k in range(numBlocks):
				if(k!=j):
					spaceAvailable[j][0]-=constraints[1][i][k][0]
			sub[j][0] = 2 * constraints[1][i][j][0] 
			sub[j][0] = sub[j][0] - spaceAvailable[j][0]
			if(j==0):
				leftMostSpace[j][0] = 0
			else:
				leftMostSpace[j][0] = leftMostSpace[j-1][0] 
				leftMostSpace[j][0] = leftMostSpace[j][0] + constraints[1][i][j-1][0]+1
			if (sub[j][0] == 0):
				continue
			if(sub[j][0] > 0):
				startingPosition = leftMostSpace[j][0] 
				startingPosition = startingPosition + spaceAvailable[j][0] 
				startingPosition = startingPosition - constraints[1][i][j][0]
				for k in range(int(startingPosition), int(startingPosition+sub[j][0])):
					solution[k][i] = 1
	for i in range(dim0):
		numBlocks = len(constraints[0][i])
		sub = np.zeros((1, numBlocks))
		spaceAvailable = np.zeros((1, numBlocks))
		leftMostSpace = np.zeros((1, numBlocks))
		for z in range(len(temp)):
			if (temp[z] == 1):
				temp[z] = 0
			elif (temp[z] == 0):
				temp[z] = 1
		for j in range (numBlocks):
			space = dim1-numBlocks+1
			spaceAvailable[0][j] = space
			for k in range(numBlocks):
				if(k!=j):
					spaceAvailable[0][j]-=constraints[0][i][k][0]
			sub[0][j] = 2 * constraints[0][i][j][0] 
			sub[0][j] = sub[0][j] - spaceAvailable[0][j]
			if(j==0):
				leftMostSpace[0][j] = 0
			else:
				leftMostSpace[0][j] = leftMostSpace[0][j-1] 
				leftMostSpace[0][j] = leftMostSpace[0][j] + constraints[0][i][j-1][0]+1
			if(sub[0][j] == 0):	
				continue
			elif(sub[0][j] > 0):
				startingPosition = leftMostSpace[0][j] 
				startingPosition = startingPosition + spaceAvailable[0][j] 
				startingPosition = startingPosition - constraints[0][i][j][0]
				for k in range(int(startingPosition), int(startingPosition+sub[0][j])):
					solution[i][k] = 1
	return solution


def findInitialCon(constraints, rowCon, colCon):
	for row in range(dim0):
		pattern = []
		currPosition = 0
		if constraints[0][row] == []:
			rowCon[row] = [np.array([0]*dim1)]			
		else:
			for block in constraints[0][row]:
				cat = block[0]+currPosition
				pattern.append(list(range(currPosition, cat)))
				currPosition+=block[0]+1
			last = dim1-len(pattern[len(pattern)-1])+1
			currentIdx = len(constraints[0][row]) -1
			rowCon[row] = helperRow(last, pattern, currentIdx, row, constraints)
	for col in range(dim1):
		currPosition = 0
		pattern = []
		if constraints[1][col] == []:
			rowCon[col] = [0 for i in range(dim0)]
		else:
			for block in constraints[1][col]:
				cat = block[0]+currPosition
				pattern.append(list(range(currPosition, cat)))
				currPosition+=block[0]+1
			last = dim0-len(pattern[len(pattern)-1])+1
			currentIdx = len(constraints[1][col])-1
			colCon[col] = helperCol(pattern, last, currentIdx, col, constraints)
	return rowCon, colCon

def helperRow(last, pattern, currentIdx, currentRow, constraints):
	tempPattern = copy.deepcopy(pattern)
	Return = []
	first = tempPattern[currentIdx][0]
	for i in range (first, last):
		if(currentIdx>0):
			next_last = i-len(tempPattern[currentIdx-1])
			arrayReturned = helperRow(next_last, tempPattern, currentIdx-1, currentRow, constraints)
			for k in arrayReturned:
				Return.append(copy.deepcopy(k))
		if(currentIdx == 0):
			newArray = np.zeros(dim1)
			for k in range(len(tempPattern)):
				for j in range(len(tempPattern[k])):
					newArray[tempPattern[k][j]] =1
			comp = solution[currentRow] - newArray
			if(all(k<=0 for k in comp)):
				Return.append(newArray)
		for j in range (len(pattern[currentIdx])):
			tempPattern[currentIdx][j]+=1
	return Return


def helperCol(pattern, last, currentIdx, currentCol, constraints):
	Return = []
	first = pattern[currentIdx][0]
	tempPattern = copy.deepcopy(pattern)
	for i in range (first, last):
		if(currentIdx>0):
			next_last = i-len(tempPattern[currentIdx-1])
			arrayReturned = helperCol(tempPattern, next_last, currentIdx-1, currentCol, constraints)
			for k in range (len(arrayReturned)):
				Return.append(arrayReturned[k])
		if(currentIdx == 0):
			newArray = np.zeros(dim0)
			for k in range(len(tempPattern)):
				for j in range(len(tempPattern[k])):
					newArray[tempPattern[k][j]] =1
			solutionCheck = np.zeros(dim0)
			for m in range(dim0):
				solutionCheck[m] = solution[m][currentCol]
			comparison = solutionCheck - newArray
			if(all(k<=0 for k in comparison)):
				Return.append(newArray)
		for j in range (len(pattern[currentIdx])):
			tempPattern[currentIdx][j]+=1
	return Return


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
