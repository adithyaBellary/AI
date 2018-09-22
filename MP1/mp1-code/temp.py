from collections import deque
import heapq

path = [ (1,2), (2,3), (4,3), (8,8), (9,9), (10,4)  ]
a = [ (1,2)]

print([x for x in path if x not in a])