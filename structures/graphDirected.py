from .graph import Graph
from .node import Node

class GraphDirected(Graph):
    def addEdge(self, edge: tuple[Node, Node], weight: float = 0) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal not in self.adj_list.keys():                # Goal node is not in the graph, prevent linking non-existant node
            return
        
        if goal not in self.adj_list[source]:               # add goal node to source node's neighbor's list + update weight
            self.adj_list[source].append(goal)
            self.weights[source][goal] = weight
        else:                                               # Edge already exists, just update the weight
            self.weights[source][goal] = weight

    def setWeight(self, edge: tuple[Node, Node], weight: float = 0) -> None:
        self.weights[edge[0]][edge[1]] = weight

    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal in self.adj_list[source]:                   # remove goal node from source node's neighbor's list
            self.adj_list[source].remove(goal)
            self.weights[source].pop(goal)