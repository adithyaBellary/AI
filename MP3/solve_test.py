import numpy as np
from copy import deepcopy
from heapq import heappush, heappop

solution = []
def solve(constraints):
    dim0 = len(constraints[0])
    dim1 = len(constraints[1])
    print("Constraints[0]")
    print(constraints[0])
    print("Constraints[1]")
    print(constraints[1])
    global solution
    solution = initialMazeReduce(constraints)
    rowSpace, colSpace = findVariableConstraintsInitial(constraints)
    print("COl Space")
    print(colSpace[0])
    solutionCheckArray = np.zeros((dim0, dim1))
    heap = []
    for i in range(0, dim0):
        thistuple = (len(rowSpace[i]), i, rowSpace[i])
        heappush(heap, thistuple)
    potentialSolution = DFS(colSpace, heap, heappop(heap),solutionCheckArray)
    print(potentialSolution)
    return potentialSolution

def findVariableConstraintsInitial(constraints):
    dim0 = len(constraints[0])
    dim1 = len(constraints[1])
    colSpace = [[] for i in range (dim1)]
    rowSpace = [[] for i in range (dim0)]
    #Iterate through each row to calculate all possible combinations for each
    for row in range(0,dim0):
        #find seed pattern
        pattern = []
        currPosition = 0
        for block in range (0, len(constraints[0][row])):
            pattern.append(list(range(currPosition, constraints[0][row][block][0]+currPosition)))
            currPosition+=constraints[0][row][block][0]+1
        rowSpace[row] =variableValuesHelperRow(pattern, pattern[len(pattern)-1][0], dim1-len(pattern[len(pattern)-1])+1, len(constraints[0][row]) -1 ,dim1, row)
    #Iterate through each column

    for col in range(0, dim1):
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
    tempPattern = deepcopy(pattern)
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
    tempPattern = deepcopy(pattern)
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

def DFS(colMtx, rowHeap, curRow, final_map):
    rowIdx = curRow[1]
    current = 0
    for row in curRow[2]:
        flag = 0
        tempFinalMap = deepcopy(final_map)
        updatedColMtx = deepcopy(colMtx)
        updatedRowHeap = deepcopy(rowHeap)
        if (len(updatedRowHeap)==0):
            tempFinalMap[rowIdx] = row
            return tempFinalMap
        for i in range(len(colMtx)):
            tempCol = []
            for j in range(len(updatedColMtx[i])):
                if(updatedColMtx[i][j][rowIdx] == row[i]):
                    tempCol.append(updatedColMtx[i][j])
            updatedColMtx[i] = tempCol
            if len(updatedColMtx[i])==0:
                flag = 1
                break
        if(flag==1):
            continue
        newRow = heappop(updatedRowHeap)
        SETROW = DFS(updatedColMtx, updatedRowHeap, newRow, tempFinalMap)
        if SETROW != []:
            SETROW[rowIdx] = row
            #print("set")
            #print(SETROW)
            return SETROW
    return []
