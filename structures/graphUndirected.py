from .graph import Graph
from .node import Node

class GraphUndirected(Graph):
    def addEdge(self, edge: tuple[Node, Node]) -> None:
        A,B = edge

        if A not in self.adj_list.keys():               # A node is not in the graph
            return
        if B not in self.adj_list.keys():               # B node is not in the graph
            return
        
        if B not in self.adj_list[A]:                   # add B node to A node's neighbor's list
            self.adj_list[A].append(B)
        if A not in self.adj_list[B]:                   # add A node to B node's neighbor's list
            self.adj_list[B].append(A)

    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        A,B = edge

        if A not in self.adj_list.keys():           # A node is not in the graph
            return
        if B in self.adj_list[A]:                   # remove B node from A node's neighbor's list
            self.adj_list[A].remove(B)

        if B not in self.adj_list.keys():           # B node is not in the graph
            return
        if A in self.adj_list[B]:                   # remove A node from B node's neighbor's list
            self.adj_list[B].remove(A)