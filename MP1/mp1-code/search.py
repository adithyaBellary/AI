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
   

# def bfs(maze):
#     # TODO: Write your code here
#     # return path, num_states_explored

#     #queue 
#     #use append and popleft to make it a queue
    
#     start = maze.getStart()
#     obj = maze.getObjectives()
#     print(obj)

#     # (node, path)

#     q = collections.deque([  (start, [start])  ])
#     v = set()

#     while True:
#         #while q is not empty
#         node, path = q.popleft()
#         # print(current)
#         #check if current if the goal state
#         #what if we have multiple goal states
#         if node in obj:
#             obj.remove(node)
#             # print(obj,"\n")
#             print(node)
#             if obj == []:
#                 break
#         if node not in v:
#             v.add(node)
#             #get list of current nodes' neighbors
#             neighbors = maze.getNeighbors(node[0], node[1])
#             for n in neighbors:
#                 temp = path + [n]
#                 q.append( (n, temp) )
    
#     print(path)
#     return path, len(v)

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    #queue 
    #use append and popleft to make it a queue
    
    start = maze.getStart()
    obj = maze.getObjectives()
    # print(obj)
    path = 0

    # (node, path)

    q = collections.deque([  (start, [start])  ])
    v = set()

    while True:
        #while q is not empty
        node, path = q.popleft()
        # print(current)
        #check if current if the goal state
        #what if we have multiple goal states
        if node in obj:
            obj.remove(node)
            # print(obj,"\n")
            # print(node)
            if obj == []:
                break
        if node not in v:
            v.add(node)
            #get list of current nodes' neighbors
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                temp = path + [n]
                q.append( (n, temp) )
    
    # print(path)
    return path, len(v)


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
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
        # print(current)
        #check if current if the goal state
        #what if we have multiple goal states
        if node in obj:
            break
        if node not in v:
            v.add(node)
            #get list of current nodes' neighbors
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                temp = path + [n]
                q.append( (n, temp) )

    return path, len(v)

def distance(a,b): 
    #print("a", a)
    #print("b", b)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    start = maze.getStart()
    obj = maze.getObjectives()

    priorityq = [(distance(start, obj[0]), start, [start])]

    #(distance to goal, node, path)
    heapq.heapify( priorityq )
    v = set()


    while priorityq:
        #get the min distance 
        dist, node, path = heapq.heappop(priorityq)
        # print(node)
        
        if node in obj:
            #we found the goal state
            break
        
        if node not in v:
            v.add(node)
            neighbors = maze.getNeighbors(node[0], node[1])
            for n in neighbors:
                temp = path + [n]
                heapq.heappush(priorityq, (distance(n, obj[0]), n, temp) )

    return path, len(v)

def h(path, node, obj):
    # print(start)
    # print(node)
    # print(obj)
    # alpha = 2
    # return distance(start, node) + alpha * distance(node, obj[0])
    # return distance(start, node) + alpha * euc(node, obj[0])
    return distance(node, obj) + len(path)

def euc(node, obj):
    return  ((node[0] - obj[0])**2 + (node[1] - obj[1])**2)**(0.5) 

# def astar(maze):
#     # TODO: Write your code here
#     # return path, num_states_explored

#     start = maze.getStart()
#     #assume we only have one objective for right now
#     obj = maze.getObjectives()

#     #(heuristic, node, path)
#     priorityq = [(h(start,start,obj), start, [start])]

#     heapq.heapify(priorityq)
#     v = set()

#     while priorityq:
#         heuristic, node, path = heapq.heappop(priorityq)

#         if node in obj:
#             #if we have reached the goal state

#             if obj.empty():
#                 break
#         if node not in v:
#             v.add(node)

#             neighbors = maze.getNeighbors(node[0], node[1])
#             for n in neighbors:
#                 temp = path + [n]
#                 heapq.heappush(priorityq, (h(path,n,obj), n, temp) )

#     return path, len(v)


def astar(maze):
    start = maze.getStart()
    obj = maze.getObjectives()

    obj_to_see = obj
    path = [start]

    h_min = min([h(start, start, obj_to_see[i]) for i in range(len(obj_to_see))])
    priorityq = [(h_min, start, path, obj_to_see)]

    v= set()

    while priorityq:
        heuristic, node, path, obj_to_see = heapq.heappop(priorityq)

        #recreate the objectives to be seen 
        OBJ = [ x for x in obj if x not in path  ]

        if (node in obj_to_see):
            obj_to_see.remove(node)

            if (len(obj_to_see) == 0):
                break

        if node not in v:
            v.add(node)
            nbrs = maze.getNeighbors(node[0], node[1])
            for n in nbrs:
                heuristic = min([h(start, n,obj_to_see[i]) for i in range(len(obj_to_see))])
                temp = path + [n]
                heapq.heappush(priorityq, (heuristic, n, temp, obj_to_see))
    print (path)
    return path, len(v)

#Approach 1: Apply Dijkstra's algorithm at each node by calling astar to find shortest path at each goal
#Approach 2: At each step, use heuristic (Manhattan) to calculate nearest goal until all are seen