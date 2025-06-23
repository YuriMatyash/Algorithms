from .graph import Graph
from .node import Node

class GraphDirected(Graph):
    def __init__(self, V: list[Node] = [], E: list[tuple] = []):
        self.flow = {}                                      #   dict{Node: dict{Node:Weight}}       <- allows access like self.flow[A][B] == flow from A to B
        super().__init__(V, E)

    def addEdge(self, edge: tuple[Node, Node], weight: float = 0, flow: float = 0) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal not in self.adj_list.keys():                # Goal node is not in the graph, prevent linking non-existant node
            return
        
        if source not in self.flow:
            self.flow[source] = {}
        if goal not in self.flow:
            self.flow[goal] = {}
        if goal not in self.flow[source]:
            self.flow[source][goal] = flow
            self.flow[goal][source] = 0                     # flow reversal support

        if goal not in self.adj_list[source]:               # add goal node to source node's neighbor's list + update weight
            self.adj_list[source].append(goal)
            self.weights[source][goal] = weight
        else:                                               # Edge already exists, just update the weight
            self.weights[source][goal] = weight

    def removeEdge(self, edge: tuple[Node, Node]) -> None:
        source, goal = edge

        if source not in self.adj_list.keys():              # Source node is not in the graph
            return
        if goal in self.adj_list[source]:                   # remove goal node from source node's neighbor's list
            self.adj_list[source].remove(goal)
            self.weights[source].pop(goal)

    def clone(self):
        newGraph = GraphDirected()
        nodeMap = {}

        nodes = self.nodes()
        edges = self.edges()

        for node in nodes:
            newNode = Node(node.value)
            nodeMap[node] = newNode
            newGraph.addNode(newNode)
        
        for edge in edges:
            left_o, right_o = edge

            left = nodeMap[left_o]
            right = nodeMap[right_o]

            weight = self.getWeight((left_o, right_o))
            flow = self.getFlow((left_o, right_o))

            newGraph.addEdge((left,right), weight=weight, flow= flow)

        return newGraph

    # Weight stuff
    ##############
    def setWeight(self, edge: tuple[Node, Node], weight: float = 0) -> None:
        self.weights[edge[0]][edge[1]] = weight

    def getWeight(self, edge: tuple[Node, Node]) -> float:
        left, right = edge
        return self.weights.get(left, {}).get(right, 0.0)

    # Flow stuff
    ##############
    def setFlow(self, edge: tuple[Node, Node], flow: float = 0) -> None:
        if edge not in self.edges():
            return

        if flow <= self.weights[edge[0]][edge[1]]:
            self.flow[edge[0]][edge[1]] = flow

    def getFlow(self, edge: tuple[Node, Node]) -> float:
        left, right = edge
        return self.flow.get(left, {}).get(right, 0.0)

    # returns how much more flow can be added
    def getResidualCapacity(self, edge: tuple[Node, Node]) -> float:
        source, target = edge
        capacity = self.weights.get(source, {}).get(target, 0.0)
        used = self.flow.get(source, {}).get(target, 0.0)
        return capacity - used