from typing import List, Tuple, Dict, Set, Optional, Any
from nptyping import NDArray
from enum import Enum

import numpy as np

class NodeType(Enum):
    SOURCE = "source"
    DESTINATION = "destination"

class TransportationNode:
    def __init__(self, type: str, max_capacity: float = None, min_capacity: float = 0):
        self.min = min_capacity
        self.max = max_capacity
        self.type = type
        self.guaranteed = None
        self.excess = None

class Solver:
    M = float('inf')
    class SolutionCell:
        def __init__(self, cost: float, value: float, chosen: bool):
            self.cost = cost
            self.value = value
            self.chosen = chosen
            self.state = None

        def __repr__(self):
            return f'({self.cost}, {self.value}, {self.chosen})'
    
    def __init__(self, sources: List[TransportationNode], destinations: List[TransportationNode], connections: List[Tuple[int, int, float]]):
        self.nodes = sources + destinations
        self.sources = sources
        self.destinations = destinations
        self.connections = connections
        self.optimalValue = None
        self.transportationMatrix = None
        self.uVector = None
        self.vVector = None
        self.destinationSubNodesValues = []
        self.sourceSubNodesValues = []

    def enhanceNodes1(self):
        '''Adds the upper bound for source nodes that didn't have it before'''

        for source in self.sources:
            if source.max is not None:
                continue
            source.max = 0
            for node in self.nodes:
                if node == source:
                    continue
                if node.type == 'destination':
                    source.max += node.max
                elif node.type == 'source':
                    source.max -= node.min
        return
    
    def enhanceNodes2(self):
        destination_sums = {}
        source_sums = {}

        for node in self.nodes:
            if node.type == 'destination':
                destination_sums[node] = destination_sums.get(node, 0) + node.max
            elif node.type == 'source':
                source_sums[node] = source_sums.get(node, 0) + node.min

        for node in self.nodes:
            if node.type == 'source' and node.max is None:
                node.max = 0
                if node in destination_sums:
                    node.max += destination_sums[node]
                if node in source_sums:
                    node.max -= source_sums[node]       
        return     

    def divideNodes(self):
        '''Divides nodes into two entities, guranateed which is the minimum required value on a node
           and excess which is the maximum value that can be transported through a node. Also added
           dummy nodes to make sure that the sum of excesses and guaranteed values are equal.'''
        
        dummyDesination = TransportationNode('dummy', None, None)
        dummySource = TransportationNode('sink', None, None)
        dummyDesination.excess, dummySource.excess = 0, 0
        tempDestinationSum, tempSourceSum = 0, 0

        for node in self.nodes:
            node.guaranteed = node.min
            node.excess = node.max - node.min
            if node.type == 'source':
                tempSourceSum += node.excess + node.guaranteed
                dummyDesination.excess += node.excess
            elif node.type == 'destination':
                tempDestinationSum += node.excess + node.guaranteed
                dummySource.excess += node.excess

        # MAKE SURE THIS REALY WORKS
        # DON'T FORGET, YOU DIDN'T CHECK

        totalDemand = tempDestinationSum + dummyDesination.excess
        totalSupply = tempSourceSum + dummySource.excess
        if totalSupply > totalDemand:
            dummyDesination.excess += totalSupply - totalDemand
        elif totalSupply < totalDemand:
            dummySource.excess += totalDemand - totalSupply


        self.destinations.append(dummyDesination)
        self.nodes.append(dummyDesination)
        self.sources.append(dummySource)
        self.nodes.append(dummySource)

        return
        
    def createNewConnections(self):
        '''Creates new connections between newly created sub-nodes 'excess' and 'guaranteed' '''

        newConnections = []
        sourceIndex, destinationIndex = 0, 0

        for connection in self.connections:
            sourceIndex, destinationIndex, cost = connection
            source, destination = self.sources[sourceIndex], self.destinations[destinationIndex]
            # Connection (Source, Destination, Cost, Flag)
            # Flag is True only if both sub-nodes have non-zero value 
            # Else it's set to false resulting in it not being added to new connectons
            placeholders = [
                (sourceIndex*2, destinationIndex*2, cost, source.guaranteed*destination.guaranteed != 0), # Guaranteed Source - Guaranteed Destination
                (sourceIndex*2, destinationIndex*2+1, cost, source.guaranteed*destination.excess != 0), # Guaranteed Source - Excess Destination
                (sourceIndex*2+1, destinationIndex*2, cost, source.excess*destination.guaranteed != 0), # Excess Source - Guaranteed Destination
                (sourceIndex*2+1, destinationIndex*2+1, cost, source.excess*destination.excess != 0), # Excess Source - Excess Destination
            ]
            newConnections.extend([newConnection[:-1] for newConnection in placeholders if newConnection[3]])

        self.connections = self.enumarateNodes(newConnections)
        return
    
    def getNumberOfNodes(self):
        '''Returns number of source nodes and destination nodes
           after dividing them into guaranteed and excess sub-nodes'''
        
        sources = set([connection[0] for connection in self.connections])
        destinations = set([connection[1] for connection in self.connections])
        return len(sources), len(destinations)
    
    def setNodeValues(self):
        for source in self.sources:
            if source.guaranteed != 0 and source.guaranteed is not None:
                self.sourceSubNodesValues.append(source.guaranteed)
            if source.excess != 0 and source.excess is not None:
                self.sourceSubNodesValues.append(source.excess)

        for destination in self.destinations:
            if destination.guaranteed != 0 and destination.guaranteed is not None:
                self.destinationSubNodesValues.append(destination.guaranteed)
            if destination.excess != 0 and destination.excess is not None:
                self.destinationSubNodesValues.append(destination.excess)
        return

    def generateDummyConnections(self, sourceMap: Dict[int, int], destinationMap: Dict[int, int]):
        '''Generates connections between dummy nodes and their respective sub-nodes
           we use the fact that indices divisble by 2 are guaranteed nodes and those that are not
           are excess nodes'''

        dummySupplyIndex = len(sourceMap)
        dummySinkIndex = len(destinationMap)
        dummyConnections = []

        for sourceIndex in sourceMap.keys():
            isExcess = sourceIndex % 2 # 0 - guaranteed, 1 - excess
            if isExcess:
                dummyConnections.append((sourceMap[sourceIndex], dummySinkIndex, 0))
            elif not isExcess:
                dummyConnections.append((sourceMap[sourceIndex], dummySinkIndex, self.M))
        
        for destinationIndex in destinationMap.keys():
            isExcess = destinationIndex % 2
            if isExcess:
                dummyConnections.append((dummySupplyIndex, destinationMap[destinationIndex], 0))
            elif not isExcess:
                dummyConnections.append((dummySupplyIndex, destinationMap[destinationIndex], self.M))

        dummyConnections.append((dummySupplyIndex, dummySinkIndex, 0))
        return dummyConnections

    def enumarateNodes(self, connections: List[Tuple[int, int, float]]):
        '''Reindexes nodes to make sure that they are enumerated from 0 to n-1
           therefore removing previously added ghost nodes
           i.e excess/guaranteed nodes that had value of 0'''
        
        sourceIndices = set()
        destinationIndices = set()
        for connection in connections:
            sourceNodeIndex = connection[0]
            destinationNodeIndex = connection[1]
            sourceIndices.add(sourceNodeIndex)
            destinationIndices.add(destinationNodeIndex)


        sourceMap = {index: i for i, index in enumerate(sourceIndices)}
        destinationMap = {index: i for i, index in enumerate(destinationIndices)}

        dummyConnections = self.generateDummyConnections(sourceMap, destinationMap)

        enumeratedConnections = [
            (sourceMap[sourceIndex], destinationMap[destinationIndex], cost)
            for sourceIndex, destinationIndex, cost in connections
        ] + dummyConnections

        return enumeratedConnections
            
    def createInitialTransportationMatrix(self):
        '''Create a transportation matrix that will be used to solve the problem'''

        height, width = self.getNumberOfNodes()
        self.transportationMatrix = np.full((height, width), fill_value=None, dtype=object)

        for connection in self.connections:
            sourceIndex, destinationIndex, cost = connection
            self.transportationMatrix[sourceIndex, destinationIndex] = self.SolutionCell(cost, 0, False)
        return
            
    def createInitialPath(self):
        '''Creates an initial path for the algorithm to start with'''

        height, width = self.getNumberOfNodes()
        y, x = 0, 0
        while x != height and y != width:
            if self.sourceSubNodesValues[y] <= self.destinationSubNodesValues[x]:
                self.transportationMatrix[y][x].value = self.sourceSubNodesValues[y]
                self.transportationMatrix[y][x].chosen = True
                self.destinationSubNodesValues[x] -= self.sourceSubNodesValues[y]
                self.sourceSubNodesValues[y] = 0
                y += 1
            elif self.sourceSubNodesValues[y] > self.destinationSubNodesValues[x]:
                self.transportationMatrix[y][x].value = self.destinationSubNodesValues[x]
                self.transportationMatrix[y][x].chosen = True
                self.sourceSubNodesValues[y] -= self.destinationSubNodesValues[x]
                self.destinationSubNodesValues[x] = 0
                x += 1
        return

    def setUVVectors(self):
        '''Sets the u and v vectors that will be used to calculate the cost of each cell.
           Starts by setting the first row/column entry to 0 that has most number of selected solution cells in it.'''
        
        height, width = self.getNumberOfNodes()
        uFilledIndices = set()
        vFilledIndices = set()

        self.uVector = np.full(height, fill_value=None, dtype=object)
        self.vVector = np.full(width, fill_value=None, dtype=object)

        startingPoint = {'row': self.uVector, 'column': self.vVector}
        chosenArray = np.array([[node.chosen for node in row] for row in self.transportationMatrix])

        maxTrueRow, noOftruths1 = np.argmax(np.sum(chosenArray, axis=1)), np.max(np.sum(chosenArray, axis=1))
        maxTrueColumn, noOftruths2 = np.argmax(np.sum(chosenArray, axis=0)), np.max(np.sum(chosenArray, axis=0))
        if noOftruths1 > noOftruths2:
            startingPoint['row'][maxTrueRow] = 0
            uFilledIndices.add(maxTrueRow)
        else:
            startingPoint['column'][maxTrueColumn] = 0
            vFilledIndices.add(maxTrueColumn)

        columnIterator = 0
        while (None in self.vVector) or (None in self.uVector):
            if columnIterator in vFilledIndices:
                selectedRows = set(np.where(chosenArray[:, columnIterator])[0])
                calculableEntries = selectedRows - uFilledIndices
                for row in calculableEntries:
                    self.uVector[row] = self.transportationMatrix[row][columnIterator].cost - self.vVector[columnIterator]
                    uFilledIndices.add(row)
                columnIterator = (columnIterator + 1)%width
                continue

            selectedRows = set(np.where(chosenArray[:, columnIterator])[0])
            calculableEntries = selectedRows.intersection(uFilledIndices)
            if not calculableEntries:
                columnIterator = (columnIterator + 1)%width
                continue

            for y in calculableEntries:
                self.vVector[columnIterator] = self.transportationMatrix[y][columnIterator].cost - self.uVector[y]
                vFilledIndices.add(columnIterator)
                columnIterator = (columnIterator + 1)%width
                continue

        return

    def setNotChosenCellsValues(self):
        '''Sets the value of cells that were not chosen to be initial path
           to values of the cell.cost[y][x] - uVector[y] - vVector[x]'''
        notChosenArray = np.array([[not node.chosen for node in row] for row in self.transportationMatrix])
        for y, x in zip(*np.where(notChosenArray)):
            self.transportationMatrix[y][x].value = self.transportationMatrix[y][x].cost - self.uVector[y] - self.vVector[x]

    def initializeProblem(self):
        '''Initializes the problem by enhancing nodes, dividing them into sub-nodes
           and creating new connections between them. Also creates the transportation matrix
           and sets the initial path for the algorithm to start with and sets the u and v vectors'''
        self.enhanceNodes1()
        self.divideNodes()
        self.createNewConnections()
        self.setNodeValues()
        self.createInitialTransportationMatrix()
        self.createInitialPath()
        self.setUVVectors()
        self.setNotChosenCellsValues()
        return      

    def findCycle(self, rowIndex: int, colIndex: int) -> dict[(int, int), (int, int)]:
        '''Finds a cycle in the transportation matrix by only traversing path cells
           that starts and ends at the cell specified by row_index and col_index'''
        
        # CHECK WHETHER THIS WORKS FOR SURE, SHAKY IMPLEMENTATION
        chosenArray = np.array([[node.chosen for node in row] for row in self.transportationMatrix])
        def dfs(y: int, x: int, startEndPos: Tuple[int, int], visited: List[List[bool]], chosenArray, path: List[Tuple[int, int]] = []) -> List[Tuple[int, int]]:
            if (y, x) == startEndPos and len(path) >= 4:
                return path.copy()

            if visited[y][x]:
                return None

            visited[y][x] = True

            # Find all possible neighbors in the same row or column
            possible_neighbors = [(index, x) for index in range(len(chosenArray)) if chosenArray[index, x]]
            possible_neighbors.extend([(y, index) for index in range(len(chosenArray[0])) if chosenArray[y, index]])

            shortest_cycle = None

            for possible_neighbor in possible_neighbors:
                if possible_neighbor == (y, x):
                    continue

                path.append((y, x))
                result = dfs(possible_neighbor[0], possible_neighbor[1], startEndPos, visited, chosenArray, path)

                if result is not None and (shortest_cycle is None or len(result) < len(shortest_cycle)):
                    shortest_cycle = result

                path.pop()

            visited[y][x] = False
            return shortest_cycle
                
            


        height, width = self.getNumberOfNodes()
        visited = [[False for _ in range(width)] for _ in range(height)]
        # print(visited)
        # print(chosenArray)
        # print(self.transportationMatrix)
        # testingArray = np.array([
        #     [True, False, False, True],
        #     [False, False, False, True],
        #     [True, False, True, False],
        #     [False, False, True, True]
        # ])
        # path = dfs(2, 2, (2, 2), visited, testingArray)
        path = dfs(rowIndex, colIndex, (rowIndex, colIndex), visited, chosenArray)
        # print(path)
        return path
    
    def extractValues(self, listOfCoordinates: List[Tuple[int, int]]) -> List[float]:
        '''Extracts the values of the cells specified by the coordinates'''
        return [self.transportationMatrix[y][x].value for y, x in listOfCoordinates]

    def solve(self):
        self.initializeProblem()
        iter = 1
        while True:
            # print(f'Iteration {iter}')
            # print(self.uVector, self.vVector)
            # print(self.transportationMatrix)
            #Selecting the cell with the most negative value
            min_negative_index = np.argmin([cell.value if (cell.value < 0 and not cell.chosen) else self.M for row in self.transportationMatrix for cell in row])
            row_index, col_index = np.unravel_index(min_negative_index, self.transportationMatrix.shape)
            if self.transportationMatrix[row_index][col_index].value >= 0:
                print('The algorithm has finished succesfully')
                print('Optimal solution found')
                self.optimalValue = sum([cell.value*cell.cost for row in self.transportationMatrix for cell in row if cell.chosen])
                break
                
            self.transportationMatrix[row_index][col_index].chosen = True
            # print(row_index, col_index, self.transportationMatrix[row_index][col_index])

            
            acceptorsDonnors = self.findCycle(row_index, col_index)
            acceptors = acceptorsDonnors[::2]
            donnors = acceptorsDonnors[1::2]
            delta = min(self.extractValues(donnors))
            self.transportationMatrix[donnors[0][0]][donnors[0][1]].chosen = False
            for node in acceptors:
                y, x = node
                self.transportationMatrix[y][x].value = max(0, self.transportationMatrix[y][x].value) + delta
            for node in donnors:
                y, x = node
                self.transportationMatrix[y][x].value = max(0, self.transportationMatrix[y][x].value) - delta

            self.setUVVectors()
            self.setNotChosenCellsValues()
            # iter += 1
            # time.sleep(1)
        return self.optimalValue, self.transportationMatrix