# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import collections
import heapq


def search(maze, searchMethod):
    return {
        "bfs": bfs(maze),
        "dfs": dfs(maze),
        "greedy": greedy(maze),
        "astar": astar(maze),
    }.get(searchMethod, [])
   

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    #queue 
    #use append and popleft to make it a queue
    
    #get the start position
    start = maze.getStart()
    #get maze objective
    obj = maze.getObjectives()
    path = 0

    # (node, path)
    q = collections.deque([  (start, [start])  ])
    v = set()

    while q:
        #while q is not empty
        node, path = q.popleft()
        #check if current if the goal state
        if node in obj:
            break
        #check if we have seen this node before
        if node not in v:
            v.add(node)
            #get list of current nodes' neighbors
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                #and the current node to the existing path
                temp = path + [n]
                q.append( (n, temp) )
    
    return path, len(v)


def dfs(maze):
    # TODO: Write your code here
    #get start position
    start = maze.getStart()
    obj = maze.getObjectives()

    #stack 
    #use append and pop to make it a stack
    # (node, path)
    q = collections.deque([  (start, [start])  ])
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

def distance(a,b): 
    #compute manhattan distance 
    #helper function
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    start = maze.getStart()
    obj = maze.getObjectives()
    #make a priority queue: heuristic is the manhattan distance
    #(heuristic, node, path)
    priorityq = [(distance(start, obj[0]), start, [start])]

    v = set()

    while priorityq:
        #pop off the element with the smallest manhattan distance
        dist, node, path = heapq.heappop(priorityq)
        if node in obj:
            #we found the goal state
            break
        #if we have not seen this node before
        if node not in v:
            v.add(node)
            #get the neighbors
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                #append this node to the path
                temp = path + [n]
                #heappush pushes the tuple and sorts on the first element,
                #which is the manhattan distance
                heapq.heappush(priorityq, (distance(n, obj[0]), n, temp) )
    return path, len(v)

def h(path, node, obj):
    return len(path) + distance(node, obj) 

# def astar_help(maze, node, obj):
#     # TODO: Write your code here
#     # return path, num_states_explored

#     start = node

#     #(heuristic, node, path)
#     priorityq = [(h([start],start,obj), start, [start])]

#     heapq.heapify(priorityq)
#     v = set()

#     while priorityq:
#         heuristic, node, path = heapq.heappop(priorityq)

#         if node == obj:
#             #if we have reached the goal state
#             break
#         if node not in v:
#             v.add(node)

#             neighbors = maze.getNeighbors(node[0], node[1])
#             for n in neighbors:
#                 temp = path + [n]
#                 heapq.heappush(priorityq, (h(path,n,obj), n, temp) )

#     return len(path) #, 

# #New Astar
# def astar(maze):
#     # for every edge, calculate and store weight in 2d list
    
#     start = maze.getStart()
#     obj_list = maze.getObjectives()
#     temp_obj_list = maze.getObjectives()

#     if len(obj_list) == 1:
        
#         #(heuristic, node, path)
#         priorityq = [(h([start],start,obj_list[0]), start, [start])]

#         heapq.heapify(priorityq)
#         v = set()

#         while priorityq:
#             heuristic, node, path = heapq.heappop(priorityq)

#             if node == obj_list[0]:
#                 #if we have reached the goal state
#                 break
#             if node not in v:
#                 v.add(node)

#                 neighbors = maze.getNeighbors(node[0], node[1])
#                 for n in neighbors:
#                     temp = path + [n]
#                     heapq.heappush(priorityq, (h(path,n,obj_list[0]), n, temp) )
#         return path, len(v)

    
#     final_path = []
#     distancelist = []

#     for row in range(len(obj_list)): distancelist += [[0]*len(obj_list)]

#     total_states_sets = set()

#     current_obj = start


#     # initialize distance list with lengths from all objectives to each other
#     for objectivei in range (len(obj_list)):
#         # calc distance from start to objective
#         temppath = astar_help(maze, start, obj_list[objectivei])
#         if objectivei == 0:
#             next_obj = obj_list[objectivei]            # objnode is next obj to go to in manhattan distance, only relevant here for start
#             minimum = temppath
#         else:
#             if (temppath < minimum):
#                 minimum = temppath
#                 next_obj = obj_list[objectivei]

#         # calculate length form obj to all other objs
#         for objectivej in range (len(obj_list)): 
#             if (objectivei == objectivej):
#                 distancelist[objectivei][objectivej] = 10000    
#             else:
#                 distancelist[objectivei][objectivej] = astar_help(maze, obj_list[objectivei], obj_list[objectivej])
        
#     for objectivei in range (len(obj_list)):
#         distancelist[objectivei][obj_list.index(next_obj)] = 10000

#     priorityq = [(h([start],start,next_obj), start, [start])]

#     heapq.heapify(priorityq)
#     v = set()
#     vtemp = set()

#     while priorityq:
#         heuristic, node, path = heapq.heappop(priorityq)

#         if len(temp_obj_list)==1: 
#             break

#         if node == next_obj:
#             if current_obj != start:
#                 temp_obj_list.remove(current_obj)
#                 obj_list[obj_list.index(current_obj)] = (-1, -1)
#             current_obj = node
#             for objectivei in range (len(obj_list)):
#                 distancelist[objectivei][obj_list.index(current_obj)] = 10000
#             next_obj = obj_list[(distancelist[obj_list.index(node)]).index(min(distancelist[obj_list.index(node)]))]
#             v = v.union(vtemp)
#             vtemp.clear()
#             priorityq = []
            
#         if node not in vtemp:
#             vtemp.add(node)
#             neighbors = maze.getNeighbors(node[0], node[1])
#             for n in neighbors:
#                 temp = path + [n]
#                 heapq.heappush(priorityq, (h(path,n,next_obj), n, temp) )
        
#     return path, len(v)

def astar_help(maze, node, obj):
    # TODO: Write your code here
    # return path, num_states_explored

    start = node

    #(heuristic, node, path)
    priorityq = [(h([start],start,obj), start, [start])]

    heapq.heapify(priorityq)
    v = set()

    while priorityq:
        heuristic, node, path = heapq.heappop(priorityq)

        if node == obj:
            #if we have reached the goal state
            break
        if node not in v:
            v.add(node)

            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                temp = path + [n]
                heapq.heappush(priorityq, (h(path,n,obj), n, temp) )

    return len(path) 

def astar(maze):
    # TODO: Write your code here    
    # for every edge, calculate and store weight in 2d list
    
    start = maze.getStart()
    obj_list = maze.getObjectives()

    priorityq = [(h([start],start,obj_list[0]), start, [start])]

    heapq.heapify(priorityq)
    v = set()

    while priorityq:
        heuristic, node, path = heapq.heappop(priorityq)

        if node in obj_list:
            #if we have reached the goal state
            break
        if node not in v:
            v.add(node)

            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                temp = path + [n]
                heapq.heappush(priorityq, (h(path,n,obj_list[0]), n, temp) )
    return path, len(v)