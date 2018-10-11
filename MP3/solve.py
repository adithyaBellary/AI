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

	# print('rowlist: ', len(rowList))
	# print('colList: ', colList[0])

	for x in range(dim0):
		# print("rowlist: ", rowList[x])
		domain = getDomain(rowList[x], dim0)
		xDomains.append((len(domain),'x', x, domain))
	
	for y in range(dim1):
		# print('colList: ', colList[y])
		domain = getDomain(colList[y], dim1)
		yDomains.append((len(domain),'y', y, domain))


	#sort the x and y domains so that we can start with the most constrained index
	heapq.heapify(xDomains)
	heapq.heapify(yDomains)
	
	# for i in xDomains:
	# 	print(i, '\n')

	
	if xDomains[0][0] < yDomains[0][0]:
		min_constrained = xDomains[0]
	else:
		min_constrained = yDomains[0]

	dfs_stack = deque(min_constrained)
	# print(dfs_stack)

	while dfs_stack:

		length, axis, idx, dom = dfs_stack.pop()

		

	

	return np.random.randint(2, size=(dim0, dim1))

	

#############################################
# 				HELPER FUNCTIONS
#############################################

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
