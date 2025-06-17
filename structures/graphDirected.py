from .graph import Graph
from .node import Node

class GraphDirected(Graph):
    def addEdge(self, edge: tuple[Node, Node]) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal not in self.adj_list.keys():                # Goal node is not in the graph, prevent linking non-existant node
            return
        if goal not in self.adj_list[source]:               # add goal node to source node's neighbor's list
            self.adj_list[source].append(goal)

    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal in self.adj_list[source]:                   # remove goal node from source node's neighbor's list
            self.adj_list[source].remove(goal)