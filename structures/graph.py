from .node import Node

# Graph is represented by an adjacency list
class Graph:
    def __init__(self, V: list[Node] = [], E: list[tuple] = []):
        self.adj_list = {}              #   dict{Node: list[Node]}
        self.weights = {}               #   dict{Node: dict{Node:Weight}}       <- allows access like self.weights[A][B] == weight from A to B
        for node in V:
            self.addNode(node)
        for edge in E:
            if len(edge) == 2:
                self.addEdge((edge[0], edge[1]))                            # default weight
            elif len(edge) == 3:
                self.addEdge((edge[0], edge[1]), weight=edge[2])            # predefined weight

    def addNode(self, node: Node) -> None:
        if node not in self.adj_list:
            self.adj_list[node] = []
            self.weights[node] = {}

    def removeNode(self, node: Node) -> None:
        self.adj_list.pop(node, None)                           # removes the node itself
        self.weights.pop(node)
        for neighborsList in self.adj_list.values():                # remove all edges going to or from the removed node
            if node in neighborsList:
                neighborsList.remove(node)
        for neighborsList in self.weights.values():
            if node in neighborsList:
                neighborsList.pop(node)

    def setWeight(self, left: Node, right: Node, weight: float) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return
    
    def getWeight(self, left: Node, right: Node, weight: float) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return

    def addEdge(self, edge: tuple[Node, Node]) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return
    
    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return

    def getChildren(self, node: Node) -> list[Node]:
        return self.adj_list.get(node, [])                          # returns list of children

    def nodes(self) -> list[Node]:
        return list(self.adj_list.keys())

    def edges(self) -> list[tuple[Node, Node]]:
        edges = []

        for currentNode in self.adj_list:
            for connectedNode in self.adj_list[currentNode]:
                edges.append((currentNode,connectedNode))

        return edges
    
    def findMaxWeight(self) -> float:
        maxWeight = 0
        for node1 in self.weights:
            for node2 in self.weights[node1]:
                if self.weights[node1][node2] > maxWeight:
                    maxWeight = self.weights[node1][node2]
        return maxWeight

    def __str__(self):
        output = ""
        for node in self.adj_list:
            neighbors = self.adj_list[node]
            neighbor_strings = []
            for neighbor in neighbors:
                neighbor_strings.append(str(neighbor))
            line = f"{node}: [{', '.join(f'{n}({self.weights[node][n]})' for n in neighbors)}]"
            output += line + "\n"
        return output.strip()