from structures.node import Node
from structures.graphDirected import GraphDirected
from structures.graphUndirected import GraphUndirected
from graphAlgorithms import *

def BFS_example():
    # Create some nodes
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    # Create a directed graph and add nodes
    graph = GraphDirected([A, B, C, D, E])

    # Add edges: A → B, A → C, C → B, B → D
    graph.addEdge((A, B))
    graph.addEdge((A, C))
    graph.addEdge((C, B))
    graph.addEdge((B, D))

    # Print graph structure
    print("Graph:")
    print(graph)

    # Run BFS starting from node A
    result = bfs(graph, A)

    # Print BFS results
    print("\nBFS Results:")
    for node, (distance, parent) in result.items():
        parent_val = parent.value if parent else None
        print(f"{node.value}: distance = {distance}, parent = {parent_val}")

def main():
    examples = {
        1: BFS_example
    }
    
    while True:
        print("What type of algorithm would you like to run an example of?")
        print("1 - BFS")
        key = input("Press key: ")
        try:
            key = int(key)
        except ValueError:
            print("Invalid input. Please enter a number.\n")
            continue

        if key in examples:
            print('\n\n#################################################################################\n')
            examples[key]()
            print('\n#################################################################################\n')
        else:
            print("Invalid key...\n")
        

if __name__ == "__main__":
    main()