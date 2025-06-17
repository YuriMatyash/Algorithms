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
        for neighbor in graph.getChildrens(currentNode):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                result[neighbor][0] = result[currentNode][0] + 1
                result[neighbor][1] = currentNode
    
    return result
            

# Depth-first search
def DFS():
    return

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

