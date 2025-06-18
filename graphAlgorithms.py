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
        newGraph.addEdge((right,left))      # flips them, that's the whole point of the transpose

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
        newGraph.addEdge((left, right))
    
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


# Strongly Connected Components
def SCC():
    return

def dijkstra():
    return

def A_star():
    return

def bellman_ford():
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

