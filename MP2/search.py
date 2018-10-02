# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

import collections
import heapq

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # TODO: Write your code here    
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
