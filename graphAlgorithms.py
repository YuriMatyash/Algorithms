import heapq    # Python's built in min heap module

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
        left = nodeMap[edge[0]]
        right = nodeMap[edge[1]]
        newGraph.addEdge((right,left), weight=graph.weights[edge[0]][edge[1]])      # flips them, that's the whole point of the transpose

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
def default_heuristic(current: Node, goal: Node, maxWeight: float = 25.0) -> float:
    distance = abs(ord(current.value) - ord(goal.value))
    return (distance / max(1.0, min(25.0, maxWeight)))


# Returns a map of all heuristic values for a graph
# returns dict{Node1:dict{Node2:distance}}
# where:    distance    -   current Node's distance to starting node
# Can then use heuristicMap[A][B] to get the value
def heuristic_map(graph: Graph, heuristicFunc: default_heuristic) -> dict[Node,dict[Node,float]]:
    heuristicMap = {}
    maxWeight = 0
    def findMaxWeight(graph: Graph):
        for node1 in graph.weights:
            for node2 in graph.weights[node1]:
                if graph.weights[node1][node2] > maxWeight:
                    maxWeight = graph.weights[node1][node2]


    nodes = graph.nodes()
    for a in nodes:
        heuristicMap[a] = {}
        for b in nodes:
            heuristicMap[a][b] = heuristicFunc(a, b)

    return heuristicMap


# A* algorithm
# WORKS ONLY WHEN VALUES OF NODES ARE ALPHABETICAL, not ascii
# finds the shortest path between two nodes
# Calculating the heuristic at runtime per edge would be better, I chose not to do that, to keep the code more readable so we just get a complete heuristic map made inside the function
def A_star(graph: Graph, start: Node, end: Node, heuristicFunc: default_heuristic):
    heuristicMap = heuristic_map(graph, heuristicFunc)

    return  # Need to make heuristic functions


def bellman_ford():
    return


def bellman_ford_rec():
    return

def DAG_shortest():
    return

def DAG_longest():
    return

# Critical Path Method
def CPM():
    return

def floyd_warshall():
    return

def ford_fulkerson():
    return

