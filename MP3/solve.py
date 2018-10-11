import numpy as np
import math
import itertools
from collections import deque
import heapq

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
	
	print('xdomains')
	for i in xDomains:
		print(i, '\n')

	print('ydomains')
	for i in yDomains:
		print(i, '\n')

	
	if xDomains[0][0] < yDomains[0][0]:
		min_constrained = xDomains[0]
	else:
		min_constrained = yDomains[0]

	dfs_stack = deque([min_constrained])

	a = [1,1,1,0]
	b = [[1,0,1,0],
		 [0,1,0,1]]

	# print(isStateAllowed(b, a))

	levelSolution = []
	
	temp_grid = np.zeros((dim0, dim1))

	levelSolution.append(temp_grid)

	# x or y, index, one possible configuration

	#how to add onto stack
	for config in xDomain[index][3]:
		dfs_stack.append((xDomains[index][1], xDomains[index][2], config))

	while dfs_stack:

		axis, current_index, current_config = dfs_stack.pop()
		print(dom)

		if isNonogramAllowed():
			break

		else:
			# Add popped tuple onto the temp grid
			new_temp_grid = levelSolution[prev_good_index] #set new tempgrid equal to old good temp grid, then add new config in to ee if it works
		
			# check is state allowed
			tempAllowed = isStateAllowed() #check if this added state is allowed 
			
			# if allowed, add new configs at unseen opposite coordinate
			if (tempAllowed):
				#check whether currently on x or y axis
				if (axis == 'x'): #xaxis
					for config in yDomain[index][3]: #change index smart, find index that hasnt been searched yet (next best constrained index in Domain)
						dfs_stack.append((yDomains[index][1], yDomains[index][2], config))
				if (axis == 'y'): #yaxis
					for config in xDomain[index][3]: #change index smart, find index that hasnt been searched yet (next best constrained index in Domain)
						dfs_stack.append((xDomains[index][1], xDomains[index][2], config))

			# else just set down the first configuration in the list of domains
			else:
				

		test = dom[0]


	return np.random.randint(2, size=(dim0, dim1))

	

#############################################
# 				HELPER FUNCTIONS
#############################################

def isStateAllowed(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing

	for con in constraints:
		inv_con = [1 if i == 0 else 0 for i in con]
		temp = [i | j for i, j in zip(inv_con, configuration)]
		#check if temp is all ones. If so, this is a valid configuration. If not, then it is not
		if 0 not in temp:
			#there is a configuration that works
			return True	

	return False

def isNonogramAllowed(constraints, configuration):
	#temp_grid: our full grid so far
	#constraint: list of lists
	#configuration: The configuration we are testing

	isStateAllowed() #iterate a bunch of times for each coordinate

	return False


def dfs(maze):
    # TODO: Write your code here
    #get start position
    start = maze.getStart()
    obj = maze.getObjectives()

    #stack 
    #use append and pop to make it a stack
    # (node, path)
    q = deque([  (start, [start])  ])
    v = set()

    while q:
        #while q is not empty
        node, path = q.pop()
        #check if the node is the goal state
        if node in obj:
            break
        if node not in v:
            v.add(node)
            #get list of current nodes' neighbors
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                #add node to the path 
                temp = path + [n]
                q.append( (n, temp) )

    return path, len(v)

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
