from .node import Node

# Graph is represented by an adjacency list
class Graph:
    def __init__(self, V: list[Node] = [], E: list[tuple[Node,Node]] = []):
        self.adj_list = {}              #   dict{Node: list[Node]}
        for node in V:
            self.addNode(node)
        for edge in E:
            self.addEdge(edge)

    def addNode(self, node: Node) -> None:
        if node not in self.adj_list:
            self.adj_list[node] = []

    def removeNode(self, node: Node) -> None:
        self.adj_list.pop(node, None)                           # removes the node itself

        for neighborsList in self.adj_list.values():                # remove all edges going to or from the removed node
            if node in neighborsList:
                neighborsList.remove(node)

    def addEdge(self, edge: tuple[Node, Node]) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return
    
    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        # Implimented in GraphDirected and GraphUndirected
        return

    def getChildrens(self, node: Node) -> list[Node]:
        return self.adj_list.get(node, [])                          # returns list of children

    def nodes(self) -> list[Node]:
        return list(self.adj_list.keys())

    def edges(self) -> list[tuple[Node, Node]]:
        edges = []

        for currentNode in self.adj_list:
            for connectedNode in self.adj_list[currentNode]:
                edges.append((currentNode,connectedNode))

        return edges
    
    def __str__(self):
        output = ""
        for node in self.adj_list:
            neighbors = self.adj_list[node]
            neighbor_strings = []
            for neighbor in neighbors:
                neighbor_strings.append(str(neighbor))
            line = f"{node}: [{', '.join(neighbor_strings)}]"
            output += line + "\n"
        return output.strip()