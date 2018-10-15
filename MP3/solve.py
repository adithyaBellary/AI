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

	xDomains = [[] for i in range(dim0)]
	yDomains = [[] for i in range(dim1)]
	print(dim0, dim1)
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
	xWeightsUnsorted, yWeightsUnsorted = calcWeights(constraints)

	xWeights = xWeightsUnsorted[np.argsort(xWeightsUnsorted[:,0])]
	yWeights = yWeightsUnsorted[np.argsort(yWeightsUnsorted[:,0])]

	levelSolution = []
	levelSolutionidx = 1
	
	temp_grid = np.zeros((dim0, dim1))
	temp_grid = temp_grid.astype(int)
	#print("temp_grid", temp_grid)
	levelSolution.append(temp_grid)

	#curr_x_idx = 0     #Current col we're checking
	#curr_y_idx = -1	   #Current row we're checking

	#Tuple (Axis, index of row/col, level index)
	seenStack = deque([])
	exploredStack = deque([])

	#Domains = (list of configurations, axis, index of row/col)
	#seenStack = (axis, index)
	if(xWeights[-1][0] <= yWeights[-1][0]):
		domain = getDomain(constraints[1][yWeights[-1][1]], dim1)
		yDomains[yWeights[-1][1]] = domain
		seenStack.append(('y', yWeights[-1][1]))
		
	else:
		domain = getDomain(constraints[0][xWeights[-1][1]], dim0)
		xDomains[xWeights[-1][1]] = domain
		seenStack.append(('x', xWeights[-1][1]))
		#exploredStack.append(('x', xWeights[-1][1], levelSolutionidx))

	xTempDomains = copy.deepcopy(xDomains)
	yTempDomains = copy.deepcopy(yDomains)

	backtrack = False
	wait_num = deque([])
	justPop = False		
	# count=0

	while seenStack:
		if (isValidSolution(constraints, levelSolution[-1]) == True):	#CHANGE HELPER FUNCTION
			print("found solution")
			break

		else:
			backtrack = False
			configAllowed = False
			axis, rowcol_index = seenStack.pop()
			print("current pop, not in solution index:",axis, rowcol_index)
			print(levelSolution[-1])
			rowcol_index = int(rowcol_index)
			while ((axis, rowcol_index) in exploredStack):
				print("index: ", rowcol_index, "was in exploredStack")
				if (isConfigAllowed(xDomains, yDomains, axis, levelSolution[-1], rowcol_index) != True):
					print("popped config not allowed")
					backtrack = True
					break
				axis, rowcol_index = seenStack.pop()
				print("current pop:",axis, rowcol_index)
				print(levelSolution[-1])
			# if((axis, rowcol_index) in exploredStack):
			# 	#in explored stack, need to check if valid, if not bt
			# 	if (isConfigAllowed(xDomains, yDomains, axis, levelSolution[-1], rowcol_index) == True):
			# 		configAllowed = True
			# 	else:
			# 		backtrack = True

			if (not backtrack): 
				#exploredStack.append((axis, rowcol_index))
				# print("ROWCOL INDEX", rowcol_index)
				if((axis == 'x') and (xDomains[rowcol_index] == [])):
					print("set x Domains for index: ", rowcol_index)
					xDomains[rowcol_index] = getDomain(constraints[0][rowcol_index], dim0)
					xTempDomains[rowcol_index] = getDomain(constraints[0][rowcol_index], dim0)
				elif((axis == 'x') and (xTempDomains[rowcol_index] == []) and (rowcol_index not in wait_num)):
					print("should not be back tracking here smh")
					xTempDomains[rowcol_index] = copy.deepcopy(xDomains[rowcol_index])
					print(wait_num)
				if((axis == 'y') and (yDomains[rowcol_index] == [])):
					print("set y Domains for index: ", rowcol_index)
					yDomains[rowcol_index] = getDomain(constraints[1][rowcol_index], dim1)
					yTempDomains[rowcol_index] = getDomain(constraints[1][rowcol_index], dim1)
				elif((axis == 'y') and (yTempDomains[rowcol_index] == []) and (rowcol_index not in wait_num)):
					print("should not be back tracking here smh")
					print(wait_num)
					yTempDomains[rowcol_index] = copy.deepcopy(yDomains[rowcol_index])
				#configAllowed = False
				#backtrack = False
				print("configAllowed", configAllowed)
				print("backtrack", backtrack)
				while (not configAllowed and not backtrack):
					# print("Rowcol index", rowcol_index)
					# print("Length of xTempDomains, yTempDomains", len(xTempDomains), len(yTempDomains))

					# print("yTempDomains", yTempDomains)
					
					if (axis == 'x'):
						print("xTempDomains", xTempDomains[rowcol_index])
						if (len(xTempDomains[rowcol_index]) == 0):
							#xTempDomains[rowcol_index] = copy.deepcopy(xDomains[rowcol_index])
							backtrack = True
						else:
							print("reaches pop xTempDomains", rowcol_index)
							currentConfig = xTempDomains[rowcol_index][-1]
							xTempDomains[rowcol_index].pop()
					if (axis == 'y'):
						print("yTempDomains", yTempDomains[rowcol_index])
						if (len(yTempDomains[rowcol_index]) == 0):
							#yTempDomains[rowcol_index] = copy.deepcopy(yDomains[rowcol_index])
							backtrack = True
						else:
							print("reaches pop yTempDomains", rowcol_index)
							currentConfig = yTempDomains[rowcol_index][-1]
							yTempDomains[rowcol_index].pop()
					#check if allowed:
					new_temp_grid = copy.deepcopy(levelSolution[-1]) #set new tempgrid equal to old good temp grid, then add new config in to ee if it works
					new_temp_grid = addConfigToState(new_temp_grid, axis, rowcol_index, currentConfig)
					# print("Temp_grid", new_temp_grid)
					configAllowed = isConfigAllowed(xDomains, yDomains, axis, new_temp_grid, rowcol_index)
					# print("Config allowed", configAllowed)
				
				if (configAllowed): #if allowed
					print("axis, index: ", axis, ", ", rowcol_index)
					print("Config allowed. currentConfig: ")
					print(currentConfig)
					print("current solution: ")
					print(new_temp_grid)
					text = input("prompt:")
					if (axis == 'x'):
						tempList = np.zeros((dim0, 2), dtype = 'int32')	
					if (axis == 'y'):						
						tempList = np.zeros((dim1, 2), dtype = 'int32')
					levelSolution.append(new_temp_grid)
					exploredStack.append((axis, rowcol_index))
					#print("Level Solution:", levelSolution[-1])
					for seen in range(len(currentConfig)):
						if (axis == 'x'):
							if currentConfig[seen] != 0:
								tempList[seen] = (copy.deepcopy(yWeightsUnsorted[seen]))
						if (axis == 'y'):
							if currentConfig[seen] != 0:
								tempList[seen] = (copy.deepcopy(xWeightsUnsorted[seen]))
					#print("before order temp list: ", tempList)
					tempList = tempList[np.argsort(tempList[:,0])]
					#print("after order Temp list: ", tempList)
					for k in tempList: # put touched indices onto stack
						if ((axis == 'x') and (k[1] != 0)):
							print("adding to seenStack: (y,", k[1],")")
							seenStack.append(('y', k[1]))
						if ((axis == 'y') and (k[1] != 0)):
							print("adding to seenStack: (x,", k[1],")")
							seenStack.append(('x', k[1]))
			# if (count > 0):
			# 	count-=1
			# else:
			print(wait_num)
			if (wait_num):# and (justPop == True)):
				wait_num.popleft()
					#justPop = False
			print(wait_num)
			if (backtrack):
				print(exploredStack)
				print(seenStack)
				#go to the last valid level solution
				# print("Backtrack:",backtrack)
				if ((axis == 'y') and ((seenStack[-1][0] != axis) or (seenStack[-1][1] != rowcol_index))):
					#yTempDomains[rowcol_index] = copy.deepcopy(yDomains[rowcol_index])
					wait_num.append(rowcol_index)
					print("appended ", rowcol_index, "to wait_num")
					print(wait_num)
					justPop = True
					print(exploredStack[-1])
					while(seenStack[-1][0]==axis):
						seenStack.pop()
					wait_num.append(exploredStack[-1][1])
					seenStack.append(exploredStack.pop())
					print(seenStack[-1])
					levelSolution.pop()
				elif ((axis == 'x') and (((seenStack[-1])[0] != axis) or ((seenStack[-1])[1] != rowcol_index))):
					#xTempDomains[rowcol_index] = copy.deepcopy(xDomains[rowcol_index])
					wait_num.append(rowcol_index)
					print("appended ", rowcol_index, "to wait_num")
					print(wait_num)
					justPop = True
					#count = 2
					while(seenStack[-1][0]==axis):
						seenStack.pop()
					wait_num.append(exploredStack[-1][1])
					seenStack.append(exploredStack.pop())
					levelSolution.pop()
				backtrack = False
			print("seenStack", seenStack)
			print("exploredStack", exploredStack)

	solutionFound = np.array(levelSolution[-1])
	#view(solutionFound)
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
	# xWeights = xWeights[np.argsort(xWeights[:,0])]
	# yWeights = yWeights[np.argsort(yWeights[:,0])]

	# print("sorted")
	# print(xWeights)
	# print(yWeights)

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

def isValidSolution(constraints, solution):
	solution = np.array(solution)
	dim0 = len(constraints[0])
	dim1 = len(constraints[1])
	if (solution.shape != (dim0, dim1)):
		print("Wrong dimensions, try flipping")
		return False
	for i in range(dim0):
		rowconstraints = constraints[0][i]
		rowcol = []
		for j in range(dim1):
			rowcol.append(solution[i][j])
		if not (runs(rowcol) == rowconstraints):
			return False
        
	for j in range(dim1):
		colconstraints = constraints[1][j]
		rowcol = []
		for i in range(dim0):
			rowcol.append(solution[i][j])
		if not (runs(rowcol) == colconstraints):
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
