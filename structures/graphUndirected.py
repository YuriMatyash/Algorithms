from .graph import Graph
from .node import Node

class GraphUndirected(Graph):
    def addEdge(self, edge: tuple[Node, Node], weight: float = 0) -> None:
        A,B = edge

        if A not in self.adj_list or B not in self.adj_list:        # One of the nodes isn't there
            return
        
        if B not in self.adj_list[A]:                   # add B node to A node's neighbor's list
            self.adj_list[A].append(B)
        if A not in self.adj_list[B]:                   # add A node to B node's neighbor's list
            self.adj_list[B].append(A)

        self.weights[A][B] = weight
        self.weights[B][A] = weight

    def setWeight(self, edge: tuple[Node, Node], weight: float = 0) -> None:
        self.weights[edge[0]][edge[1]] = weight
        self.weights[edge[1]][edge[0]] = weight

    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        A,B = edge

        if A not in self.adj_list or B not in self.adj_list:        # One of the nodes isn't there
            return
        
        if B in self.adj_list[A]:                   # remove B node from A node's neighbor's list
            self.adj_list[A].remove(B)
            self.weights[A].pop(B)

        if A in self.adj_list[B]:                   # remove A node from B node's neighbor's list
            self.adj_list[B].remove(A)
            self.weights[B].pop(A)