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


def SCC_example():
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    graph = GraphDirected([A, B, C, D, E], [
        (A, B), (B, C), (C, A),  # SCC: A, B, C
        (B, D),
        (D, E), (E, D)           # SCC: D, E
    ])

    print("Graph:")
    print(graph)
    print("\nSCCs:")
    sccs = SCC(graph)
    for i, scc in enumerate(sccs, 1):
        print(f"{i}: {[str(n) for n in scc]}")


def Dijkstra_example():
    # helper function to show the final paths
    def reconstruct_path(result, target):
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = result[current][1]
        return list(reversed(path))
    
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    graph = GraphDirected(
        [A, B, C, D, E],
        [
            (A, B, 1),
            (A, C, 4),
            (B, C, 2),
            (B, D, 5),
            (C, D, 1),
            (D, E, 3)
        ]
    )
    print("Graph:")
    print(graph)
    result = dijkstra(graph, A)

    print("\nDijkstra result (from A):")
    for node, (dist, parent) in result.items():
        parent_val = parent.value if parent else None
        print(f"{node}: distance = {dist}, parent = {parent_val}")

    # reconstructs path from A to E
    target = E
    path = reconstruct_path(result, target)
    print(f"\nShortest path from A to {target}: {' -> '.join(str(n) for n in path)}")


def A_star_example():
    A = Node("A", (0, 0))
    B = Node("B", (1, 0))
    C = Node("C", (2, 0))
    D = Node("D", (1, 1))
    E = Node("E", (2, 1))

    graph = GraphDirected([A, B, C, D, E])
    graph.addEdge((A, B), weight=1)
    graph.addEdge((B, C), weight=1)
    graph.addEdge((C, E), weight=1)
    graph.addEdge((A, D), weight=2)
    graph.addEdge((D, E), weight=2)

    print("Graph:")
    print(graph)
    print("\nRunning A* from A to E...\n")

    path, cost = A_star(graph, A, E)

    if path:
        print("Shortest Path Found:")
        print(" → ".join(str(node) for node in path))
        print(f"Total Cost: {cost}")
    else:
        print("No path found from A to E.")


def bellman_ford_example():
    A = Node("A", (0, 0))
    B = Node("B", (1, 0))
    C = Node("C", (2, 0))
    D = Node("D", (1, 1))
    graph = GraphDirected([A, B, C, D])
    graph.addEdge((A, B), 1)
    graph.addEdge((B, C), 3)
    graph.addEdge((A, D), 4)
    graph.addEdge((D, C), -2)  # Negative weight, still no negative cycle

    print("Graph:")
    print(graph)

    print("\nRunning Bellman-Ford from A...\n")
    result = bellman_ford(graph, A)

    if result is None:
        print("Negative weight cycle detected. Aborting.")
    else:
        distances, parents = result
        print("Shortest distances from A:")
        for node in graph.nodes():
            print(f"  {node}: {distances[node]}")
        print("\nPaths (via parents):")
        for node in graph.nodes():
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()
            print(f"  {node}: {' → '.join(str(n) for n in path)}")


def DAG_shortest_example():
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    graph = GraphDirected([A, B, C, D, E], [
        (A, B, 3),
        (A, C, 6),
        (B, C, 4),
        (B, D, 4),
        (C, D, 8),
        (C, E, 11),
        (D, E, -4)
    ])

    print("Graph:")
    print(graph)
    print("\nRunning DAG Shortest Paths from A...\n")

    result = DAG_shortest(graph, A)

    if result is None:
        print("The graph is not a DAG.")
        return

    distances, parents = result

    print("Shortest distances from A:")
    for node, dist in distances.items():
        print(f"  {node}: {dist}")

    print("\nPaths (via parents):")
    for node in graph.nodes():
        path = []
        current = node
        while current:
            path.append(current)
            current = parents[current]
        print(f"  {node}: {' → '.join(str(n) for n in reversed(path))}")


def DAG_longest_example():
    A = Node('A')
    B = Node('B')
    C = Node('C')
    D = Node('D')
    E = Node('E')

    graph = GraphDirected([A, B, C, D, E], [
        (A, B, 3),
        (A, C, 6),
        (B, C, 4),
        (B, D, 4),
        (C, D, 8),
        (D, E, 2)
    ])

    print("Graph:")
    print(graph)
    print("\nRunning DAG_longest from A...")

    result, parent = DAG_longest(graph, A)

    print("\nLongest distances from A:")
    for node in graph.nodes():
        print(f"  {node}: {result[node]}")

    print("\nPaths (via parents):")
    for node in graph.nodes():
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        print(f"  {node}: {' → '.join(str(n) for n in path)}")


def CPM_example():
    graph = GraphDirected()
    A = Node('A')
    B = Node('B')
    C = Node('C')
    D = Node('D')

    graph.addNode(A)
    graph.addNode(B)
    graph.addNode(C)
    graph.addNode(D)

    graph.addEdge((A, B), 3)
    graph.addEdge((A, C), 2)
    graph.addEdge((B, D), 4)
    graph.addEdge((C, D), 1)

    finish_times, father = CPM(graph)

    print("Longest Finish Times:")
    for node in finish_times:
        print(f"{node.value}: {finish_times[node]}")

    critical_end = max(finish_times, key=finish_times.get)
    critical_path = []

    while critical_end is not None:
        critical_path.append(critical_end)
        critical_end = father[critical_end]

    critical_path = critical_path[::-1]

    print("\nCritical Path:")
    print(" -> ".join(node.value for node in critical_path))


def main():
    examples = {
        1: BFS_example,
        2: DFS_example,
        3: edge_classification_example,
        4: topological_sort_example,
        5: SCC_example,
        6: Dijkstra_example,
        7: A_star_example,
        8: bellman_ford_example,
        9: DAG_shortest_example,
        10: DAG_longest_example,
        11: CPM_example
    }
    
    while True:
        print("What type of algorithm would you like to run an example of?")
        print(
            "1 - Graph - BFS" \
            "\n2 - Graph - DFS"\
            "\n3 - Directed Graph - Edge Classification"\
            "\n4 - Directed Acyclic Graph - Topological Sort"\
            "\n5 - Directed Graph - SCC"\
            "\n6 - Positive Weighted Graph - Dijkstra"\
            "\n7 - Positive Weighted Graph - A*"\
            "\n8 - Directed Weighted Graph - Bellman-Ford"\
            "\n9 - Acyclic Directed Weighted Graph - DAG Single-Source Shortest Path"\
            "\n10 - Acyclic Directed Weighted Graph - DAG Single-Source Longest Path"\
            "\n11 - Acyclic Directed Weighted Graph - CPM"
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