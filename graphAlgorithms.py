import heapq    # Python's built in min heap module
from typing import Callable     # Allows calling of functions

from structures.node import Node
from structures.graph import Graph
from structures.graphDirected import GraphDirected
from structures.graphUndirected import GraphUndirected

# Returns a transposed graph
def transpose(graph: GraphDirected) -> GraphDirected:
    newGraph = GraphDirected()
    nodeMap = {}                           # Since I want to make copies, and not reuse the same Node objects, I later have a problem adding edges that's why i need to map

    nodes = graph.nodes()
    edges = graph.edges()

    for node in nodes:
        newNode = Node(node.value)          # Make a copy so to not use the same object
        nodeMap[node] = newNode
        newGraph.addNode(newNode)
    
    for edge in edges:
        left_o, right_o = edge

        left = nodeMap[left_o]
        right = nodeMap[right_o]

        weight = graph.getWeight(left_o, right_o)
        flow = graph.getFlow((left_o, right_o))

        newGraph.addEdge((right,left), weight=weight, flow=flow)      # flips them, that's the whole point of the transpose

    return newGraph


# Returns an Undirected graph from a directed graph
def removeDirections(graph: GraphDirected) -> GraphUndirected:
    newGraph = GraphUndirected()
    nodeMap = {}

    nodes = graph.nodes()
    edges = graph.edges()

    for node in nodes:
        newNode = Node(node.value)          # Make a copy so to not use the same object
        nodeMap[node] = newNode
        newGraph.addNode(newNode)
    
    for edge in edges:
        left = nodeMap[edge[0]]
        right = nodeMap[edge[1]]
        newGraph.addEdge((left, right), weight = 0)         # Makes no sense to keep weights, if there's an edge both way which weight would we keep
    
    return newGraph


# Breadth-First Search
# returns dict{Node: list[d, pi]}
# where:    d   -   distance from start
#           pi  -   Node's parent
def BFS(graph: Graph, start: Node) -> dict[Node, list]:
    result = {}
    queue = []
    visited = set()             # Better than a list, o(1) access instead of o(n)

    for node in graph.nodes():
        result[node] = [float('inf'), None]         # [distance, parent]
    
    result[start][0] = 0
    queue.append(start)
    visited.add(start)
    
    while queue:
        currentNode = queue.pop(0)
        for neighbor in graph.getChildren(currentNode):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                result[neighbor][0] = result[currentNode][0] + 1
                result[neighbor][1] = currentNode
    
    return result


# Returns True if Undirected Graph is connected
def isConnected(graph: GraphDirected) -> bool:
    nodes = graph.nodes
    
    if len(nodes) <= 1:             # Base case, no nodes/one node -> trivially connected
        return True      

    # Undirected Grapgh - connected if all nodes reachable from any start node
    startNode = graph.nodes()[0]
    BFS_result = BFS(graph,startNode)

    for node in graph.nodes():
        if BFS_result[node][0] == float('inf'):     # Node wasn't reached during BFS
            return False
    return True


# Returns True if a directed graph is weakly connected
# It's weakly connected if it's undirected equivilent is connected
def isWeaklyConnected(graph: GraphDirected) -> bool:
    unDirected = removeDirections(graph)
    if isConnected(unDirected):
        return True
    return False


# Returns True if directed graph is strongly connected
# It's strongly connected if from every node in the graph we can reach all other nodes
# It's enough to check on a single node in the original graph, and in the transposed graph
def isStronglyConnected(graph: GraphDirected) -> bool:
    nodes = graph.nodes
    if len(nodes) <= 1:             # Base case, no nodes/one node -> trivially connected
        return True  
    
    startNode = nodes[0]
    BFS_result = BFS(graph, startNode)
    for node in graph.nodes():
        if BFS_result[node][0] == float('inf'):     # Node wasn't reached during BFS
            return False
        
    graphT = transpose(graph) 
    BFS_T_result = BFS(graphT, startNode)   
    for node in graph.nodes():
        if BFS_T_result[node][0] == float('inf'):     # Node wasn't reached during BFS
            return False
        
    return True


# Depth-first search
# returns dict{Node: list[d, f, pi]}
# where:    d   -   discovery time
#           f   -   finish time
#           pi  -   Node's parent
def DFS(graph: Graph) -> dict[Node, list]:
    color = {}
    result = {}
    time = [0]          # can't use int, python makes it local to DFS, can't use in DFS_visit

    for node in graph.nodes():
        result[node] = [0,0, None]
        color[node] = "white"

    def DFS_visit(currentNode: Node):
        color[currentNode] = "gray"
        time[0] += 1
        result[currentNode][0] = time[0]                    # first pass(d array)
        for neighbor in graph.getChildren(currentNode):
            if color[neighbor] ==  "white":
                result[neighbor][2] = currentNode           # set node's parent(pi array)
                DFS_visit(neighbor)
        color[currentNode] = "black"
        time[0] += 1
        result[currentNode][1] = time[0]                    # final pass(f array)

    for node in graph.nodes():
        if color[node] == "white":
            DFS_visit(node)

    return result


# Classify edges of a directed graph
# returns dict{str: list[tuple(Node1,Node2)]}
# where:    str     -   type of edge(tree,forward,back,cross)
#           Node1   -   source node
#           Node2   -   target node
def classify_edges(graph: GraphDirected) -> dict[str, list]:
    d,f,pi = {},{},{}
    result = {
        "tree":[],              #   node2 is child of node1
        "forward":[],           #   d[node1] < d[node2] < f[node2] < f[node1]   &&  not tree edge
        "back":[],              #   d[node2] < d[node1] < f[node1] < f[node2]
        "cross":[]              #   f[noe2] < d[node1]
    }
    DFS_result = DFS(graph)
    for node in DFS_result:
        d[node] = DFS_result[node][0]
        f[node] = DFS_result[node][1]
        pi[node] = DFS_result[node][2]

    for node1 in graph.nodes():
        for node2 in graph.getChildren(node1):
            if pi[node2] == node1:
                result["tree"].append((node1,node2))
            elif d[node1] < d[node2] and f[node2] < f[node1]:
                result["forward"].append((node1,node2))
            elif d[node2] < d[node1] and f[node1] < f[node2]:
                result["back"].append((node1,node2))
            elif f[node2] < d[node1]:
                result["cross"].append((node1,node2))
    
    return result


# Checks if a graph is a DAG - Directed Acyclic Graph
# if during a DFS we detect a back edge, there's a cycle inside the graph -> It isn't DAG
# Because I don't want to just rewrite DFS with edge classification, I'll just use classify_edges
# It has worse run time, but the code is much cleaner and the input isn't going to be that big anyway
def isDAG(graph: GraphDirected) -> bool:
    result = classify_edges(graph)

    if result["back"] != []:
        return False
    return True


# Makes a topological sort using DFS
# in the returned stack the sort is from left to right
# if in the grab A->B
# Then the resulting stack will have [A,B]
# If the graph is not DAG then it returns None(No valid topological sort)
def topological_sort(graph: GraphDirected) -> list[Node]:
    if not isDAG(graph):
        return None
    
    color = {}
    result = []

    for node in graph.nodes():
        color[node] = "white"

    def DFS_visit(currentNode: Node):
        color[currentNode] = "gray"
        for neighbor in graph.getChildren(currentNode):
            if color[neighbor] ==  "white":
                DFS_visit(neighbor)
        color[currentNode] = "black"
        result.append(currentNode)

    for node in graph.nodes():
        if color[node] == "white":
            DFS_visit(node)
    
    result.reverse()
    return result


# Finds all Strongly Connected Components in a Directed graph
# Uses Kosaraju’s Algorithm.
# 1. DFS on original graph
# 2. Compute transpose of graph
# 3. DFS_connected on transposed graph in reverse order of finishing times
# 4. each part is an SCC
def SCC(graph: GraphDirected) -> list[list[Node]]:
    result = []
    color = {}
    DFS_stack = []

    # 1. Basic DFS on entire graph just to store stack sorted by finishing times
    for node in graph.nodes():
        color[node] = "white"

    def DFS_visit(currentNode: Node):
        color[currentNode] = "gray"
        for neighbor in graph.getChildren(currentNode):
            if color[neighbor] ==  "white":
                DFS_visit(neighbor)
        color[currentNode] = "black"
        DFS_stack.append(currentNode)

    for node in graph.nodes():
        if color[node] == "white":
            DFS_visit(node)

    # 2 . Transpose of original graph
    graph_T = transpose(graph)
    
    # 3+4 DFS and add to the SCC list
    for node in graph_T.nodes():
        color[node] = "white"

    def DFS_connected(node: Node, currentSCC: list[Node]):
        color[node] = "gray"
        currentSCC.append(node)
        for neighbor in graph_T.getChildren(node):
            if color[neighbor] == "white":
                DFS_connected(neighbor, currentSCC)
        color[node] = "black"

    while DFS_stack:
        node = DFS_stack.pop()
        if color[node] == "white":
            currentSCC = []
            DFS_connected(node, currentSCC)
            result.append(currentSCC)

    return result


# Dijkstra algorithm
# Finds the distances of all nodes to a specific node uses a min heap
# returns dict{Node:tuple(distance,Parent)}
# where:    distance    -   current Node's distance to starting node
#           Parent      -   current Node's parent
def dijkstra(graph: Graph, start: Node) -> dict[Node,tuple[float,Node]]:
    result = {}
    for node in graph.nodes():                           # Initialization, each node gets distance from start as inf, father is None
        result[node] = (float('inf'), None)
    result[start] =  (0, None)
    
    minHeap = [(0, start)]                          # (distance, Node)
    visited = set()

    while minHeap:      # Goes over all nodes in the graph(which are connected to start)
        distance, currentNode = heapq.heappop(minHeap)    # current node's distance from start, and father node

        if currentNode in visited:                         # Already visited this node previously
            continue
        visited.add(currentNode)

        if distance == float('inf'):                # All remaining nodes are unreachable
            break  

        for childNode in graph.getChildren(currentNode):            # Go over all the children of the current node
            weight = graph.weights[currentNode][childNode]          # weight to go from current to child
            newRoute = distance + weight                            # weight to go from start to child
            # RELAX function, if the child's distance is larger, update it.
            if newRoute < result[childNode][0]:                     # if the new weight to go from start to child is smaller than the previously thought weight      
                result[childNode] = (newRoute, currentNode)         # update the weight
                heapq.heappush(minHeap, (newRoute, childNode))      # If updated it(we always update atleast once if connected), add to the heap

    return result


# Heuristic function to make A* work
# maxWeight - the max weight in the graph
# The idea is to take the numeric difference between letters
# Then divide that value by either 25 or the highest value of weight, the lower one between the two
# If it's smaller than 1, divide by 1 instead(so to not increase the heuristic value)
# tries to Keeps heuristic smaller than actual weights, else A* won't work
def heuristic_value(current: Node, goal: Node, maxWeight: float = 25) -> float:
    distance = abs(ord(current.value) - ord(goal.value))
    return (distance / max(1.0, min(25.0, maxWeight)))


# Standard euclidean with scaling
def heuristic_euclidean(current: Node, goal: Node, maxWeight: float = 1) -> float:
    x1, y1 = current.coordinates
    x2, y2 = goal.coordinates
    distance = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
    return distance / max(1.0, maxWeight)


# Standard manhatan with scaling
def heuristic_manhattan(current: Node, goal: Node, maxWeight: float = 1) -> float:
    x1, y1 = current.coordinates
    x2, y2 = goal.coordinates
    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance / max(1.0, maxWeight)


# Standard chebyshev
def heuristic_chebyshev(current: Node, goal: Node, maxWeight: float = 1) -> float:
    x1, y1 = current.coordinates
    x2, y2 = goal.coordinates
    distance = max(abs(x1 - x2), abs(y1 - y2))
    return distance / max(1.0, maxWeight)


# Returns a map of all heuristic values for a graph
# returns dict{Node1:dict{Node2:distance}}
# where:    distance    -   current Node's distance to starting node
# Can then use heuristicMap[A] to get the value to the end node
def heuristic_map(graph: Graph, end: Node, heuristicFunc: Callable[[Node, Node, float], float] = heuristic_euclidean) -> dict[Node,float]:
    heuristicMap = {}
    maxWeight = graph.findMaxWeight()

    for current in graph.nodes():
        heuristicMap[current] = heuristicFunc(current, end, maxWeight)

    return heuristicMap


# A* algorithm
# IF NO HEURISTIC FUNCTION IS PASSED - IT WORKS ONLY WHEN VALUES OF NODES ARE ALPHABETICAL
# finds the shortest path between two nodes
# Calculating the heuristic at runtime per edge would be better, I chose not to do that, to keep the code more readable so we just get a complete heuristic map made inside the function
# Calculating at runtime is how A* ACTUALLY works, but I care more about readability and changing it to work at runtime isn't that much different
# The algorithm is remade through the pseudocode we were presented in class, with some slight modifications
def A_star(graph: Graph, start: Node, end: Node, heuristicFunc: Callable[[Node, Node, float], float] = heuristic_euclidean) -> tuple[list[Node], float]:
    heuristicMap = heuristic_map(graph, end, heuristicFunc)
    nodes = graph.nodes()
    costFromStart = {}              # Actual shortest cost from the start to a given node
    father = {}                     # Current node's father according to A*, used for backtracking
    costEstimateToGoal = {}         # Estimated total cost from start to goal through a node

    for node in nodes:
        costFromStart[node] = float('inf')
        costEstimateToGoal[node] = float('inf')
        father[node] = None
    costFromStart[start] = 0
    costEstimateToGoal[start] = heuristicMap[start]
    father[start] = None

    Queue = []              # Current possible nodes to visit
    heapq.heappush(Queue, (costEstimateToGoal[start], start))   
    visited = set()         # Nodes already visited

    while Queue:
        _, current = heapq.heappop(Queue)   # costEestimateToGoal, node
        visited.add(current)

        # finished, go over the fathers and return the shortest path + it's weight
        if current == end:
            path = []
            while current:                  # fills path from end to start
                path.append(current)
                current = father[current]
            return (list(reversed(path)), costFromStart[end])
        
        for neighbor in graph.getChildren(current):
            # Node was already previouly visited
            if neighbor in visited:
                continue

            cost = costFromStart[current] + graph.weights[current][neighbor]
            if cost < costFromStart[neighbor]:
                costFromStart[neighbor] = cost
                father[neighbor] = current
                costEstimateToGoal[neighbor] = cost + heuristicMap[neighbor]

            heapq.heappush(Queue, (costEstimateToGoal[neighbor], neighbor))

    return (None, float('inf'))                 # Failed to reach end Node from start Node


# Bellman Ford algroithm
# Returns the shortest path to all nodes from a starting node
# Works on Negative graphs! :)
def bellman_ford(graph: GraphDirected, start: Node) -> tuple[dict[Node,float],dict[Node,Node]]:
    result = {}             # Node:cost to start
    father = {}             # Node:father
    nodes = graph.nodes()
    edges = graph.edges()

    for node in nodes:
        result[node] = float('inf')
        father[node] = None
    result[start] = 0

    # RELAX edges repeatedly
    for _ in range(len(nodes) - 1):
        for edge in edges:      # edge is (u,v)
            u = edge[0]
            v = edge[1]
            weight = graph.weights[u][v]
            if (result[u] + weight < result[v]):
                result[v] = result[u] + weight
                father[v] = u
    
    # Check for negative cycles
    # Basically do it all over, if you need to update, means there's a negative cycle
    # You need V-1 iteration at most to finish, if you still update on the Vth iteration -> there's a negative cycle
    for edge in edges:
        u = edge[0]
        v = edge[1]
        weight = graph.weights[u][v]
        if (result[u] + weight < result[v]):            # negative cycle 
            return None                       
        
    return result, father


# DAG-Shortest Paths
# An algorithm that is used to return all the shortest paths from a single node to all other nodes
# Only works on a weighted acyclic directed graph 
def DAG_shortest(graph: GraphDirected, start: Node) -> tuple[dict[Node,float],dict[Node,Node]]:
    if not isDAG(graph):            # Not acyclic
        return None
    
    result = {}
    father = {}
    nodes = graph.nodes()
    edges = graph.edges()
    topoSorted = topological_sort(graph)

    # They call it: INITIALIZE-SINGLE-SOURCE(G,s)
    # They put these 4 lines in a function....
    for node in nodes:
        result[node] = float('inf')
        father[node] = None
    result[start] = 0

    # RELAX
    for node in topoSorted:
        for neighbor in graph.getChildren(node):
            weight = graph.weights[node][neighbor]
            if result[neighbor] > result[node] + weight:
                result[neighbor] = result[node] + weight
                father[neighbor] = node

    return  result, father


# DAG-Longest Paths
# An algorithm that is used to return all the longest paths from a single node to all other nodes
# Only works on a weighted acyclic directed graph 
def DAG_longest(graph: GraphDirected, start: Node) -> tuple[dict[Node,float],dict[Node,Node]]:
    if not isDAG(graph):            # Not acyclic
        return None
    
    result = {}
    father = {}
    nodes = graph.nodes()
    edges = graph.edges()
    topoSorted = topological_sort(graph)

    for node in nodes:
        result[node] = -float('inf')
        father[node] = None
    result[start] = 0

    # RELAX
    for node in topoSorted:
        for neighbor in graph.getChildren(node):
            weight = graph.weights[node][neighbor]
            if result[neighbor] < result[node] + weight:
                result[neighbor] = result[node] + weight
                father[neighbor] = node

    return result, father


# Returns a list of nodes which no one points to(start nodes)
# Used by CPM
def starting_nodes(graph: Graph) -> list[Node]:
    pointedToCount = {}
    startNodes = []
    nodes = graph.nodes()

    for node in nodes:
        pointedToCount[node] = 0
    
    # Count how many times each node is a child of another node
    for node in nodes:
        children = graph.getChildren(node)
        for neighbor in children:
            pointedToCount[neighbor] += 1
    
    # if a node is never a child -> it's a starting node
    for key in pointedToCount:
        if pointedToCount[key] == 0:
            startNodes.append(key)

    return startNodes


# Critical Path Method
# Finds the longest path from start to finish based on weights that represent task times to finish
# It's basically a fancier and more global version of the DAG longest path method
def CPM(graph: Graph) -> tuple[dict[Node,float],dict[Node,Node]]:
    if not isDAG(graph):            # Not acyclic
        return None
    
    finish = {}             # longest finish time for each node
    father = {}             # parent of finished node

    startingNodes = starting_nodes(graph)   

    # Add an artifical Node with edges of weight 0 to all starting nodes
    artificalNode = Node('ArtificalNode')
    graph.addNode(artificalNode)
    for node in startingNodes:
        graph.addEdge((artificalNode, node), 0)

    nodes = graph.nodes()
    edges = graph.edges()
    topoSorted = topological_sort(graph)

    for node in nodes:
        finish[node] = 0
        father[node] = None

    # RELAX
    for node in topoSorted:
        for neighbor in graph.getChildren(node):
            weight = graph.weights[node][neighbor]
            if finish[neighbor] < finish[node] + weight:
                finish[neighbor] = finish[node] + weight
                father[neighbor] = node

    return  finish, father


# Floyd Warshall algorithm
# Works for weights, directed graphs with negative weights
# Finds all shortest paths for every pair of nodes
# impressive stuff
def floyd_warshall(graph: Graph) -> dict[Node, dict[Node, float]]:
    result = {}
    nodes = graph.nodes()

    # initialize first keys
    for node in nodes:
        result[node] = {}

    # initialize complete result dict
    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:  
                result[node1][node2] = 0
            elif node2 in graph.getChildren(node1):
                result[node1][node2] = graph.weights[node1][node2]
            else:
                result[node1][node2] = float('inf')

    # main triple loop
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if result[i][j] > result[i][k] + result[k][j]:
                    result[i][j] = result[i][k] + result[k][j]

    # How to find negative cycles once algorithm finishes
    '''    
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if result[node][node] < 0:
                    # Negative cycle is present
    '''

    return result


# Ford fulkerson algorithm
def ford_fulkerson():
    return

