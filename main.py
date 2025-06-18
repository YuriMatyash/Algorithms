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
    result = BFS(graph, A)

    # Print BFS results
    print("\nBFS Results:")
    for node, (distance, parent) in result.items():
        parent_val = parent.value if parent else None
        print(f"{node.value}: distance = {distance}, parent = {parent_val}")


def DFS_example():
    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")
    e = Node("E")

    graph = GraphDirected([a, b, c, d, e], [
        (a, b),
        (a, c),
        (b, d),
        (c, d),
        (d, e)
    ])

    print("Graph:")
    print(graph)

    # Run DFS
    result = DFS(graph)

    print("\nDFS result:")
    for node in result:
        d, f, pi = result[node]
        parent = pi.value if pi else None
        print(f"{node.value}: d={d}, f={f}, pi={parent}")


def edge_classification_example():
    a = Node("A")
    b = Node("B")
    c = Node("C")
    d = Node("D")
    e = Node("E")

    graph = GraphDirected([a, b, c, d, e], [
        (a, b),
        (a, c),
        (b, d),
        (c, d),
        (d, e),
        (e, b)   # back edge
    ])

    print("Graph:")
    print(graph)

    edge_types = classify_edges(graph)

    print("\nEdge Classification:")
    for edge_type in edge_types:
        print(f"{edge_type.upper()} edges:")
        for u, v in edge_types[edge_type]:
            print(f"  {u.value} → {v.value}")
        print()


def topological_sort_example():
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    graph = GraphDirected(
        V=[A, B, C, D, E],
        E=[(A, C),(B, C),(C, D),(D, E)]
    )

    print("Graph:")
    print(graph)
    print()

    topo_order = topological_sort(graph)
    print("Topological Sort:")
    print(" → ".join(str(node) for node in topo_order))


def main():
    examples = {
        1: BFS_example,
        2: DFS_example,
        3: edge_classification_example,
        4: topological_sort_example
    }
    
    while True:
        print("What type of algorithm would you like to run an example of?")
        print(
            "1 - Graph - BFS" \
            "\n2 - Graph - DFS"\
            "\n3 - Directed Graph - Edge Classification"\
            "\n4 - Directed Acyclic Graph - Topological Sort"
        )
        print("TO EXIT WRITE 0")
        key = input("Press key: ")
        try:
            key = int(key)
        except ValueError:
            print("Invalid input. Please enter a number.\n")
            continue
        if key == 0:
            break

        if key in examples:
            print('\n\n#################################################################################\n')
            examples[key]()
            print('\n#################################################################################\n')
        else:
            print("Invalid key...\n")
        

if __name__ == "__main__":
    main()