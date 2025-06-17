from structures.node import Node
from structures.graph import Graph
from structures.graphDirected import GraphDirected
from structures.graphUndirected import GraphUndirected

# Breadth-First Search
# returns dict{Node: list[d, pi]}
# where:    d   -   distance from start
#           pi  -   Node's parent
def bfs(graph: Graph, start: Node) -> dict[Node, list]:
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

def topological_sort():
    return

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

