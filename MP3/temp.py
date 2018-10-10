import numpy as np
import math
import itertools
from collections import deque
def getDomain(constraints, dimension=5):

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

a = [[1, 1], [2, 2], [1, 3]]
d = 5
ret = getDomain(a,d)

# print(ret)
x = [1,2,3]

masterList = []

colors = [1,1,1]
q = deque([colors])
zeros = [0,0]
while q:
    #while the queue is empty
    arr = q.pop()
    if len(arr) == (len(colors) + len(zeros)):
        if (isValid(arr)) and (arr not in masterList):
            masterList.append(arr)
    else:
        #insert the zeros
        for i in range(len(arr) + 1):
            temp = arr.copy()
            temp.insert(i, 0)
            q.append(temp)

