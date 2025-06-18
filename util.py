from structures.node import Node
from structures.graph import Graph
from structures.graphDirected import GraphDirected
from structures.graphUndirected import GraphUndirected
from graphAlgorithms import classify_edges, DFS, BFS

# Checks if a graph is a DAG - Directed Acyclic Graph
# if during a DFS we detect a back edge, there's a cycle inside the graph -> It isn't DAG
# Because I don't want to just rewrite DFS with edge classification, I'll just use classify_edges
# It has worse run time, but the code is much cleaner and the input isn't going to be that big anyway
def isDAG(graph: GraphDirected) -> bool:
    result = classify_edges(graph)

    if result["back"] != []:
        return False
    return True


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
        newGraph.addEdge((right,left))      # flips them, that's the whole point of the transpose

    return newGraph


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
        newGraph.addEdge((left, right))
    
    return newGraph


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