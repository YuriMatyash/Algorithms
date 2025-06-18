# Algorithms
A collection of algorithm learnt during Algorithms 1+2 at H.I.T(Holon's Institute of Technology)<br>
All implemented in python for the sake of learning and deep understanding

## Algorithms covered

There are `util functions` and there are the main algorithms<br>
All of these are covered inside `graphAlgorithms.py`

- `transpose()`                 - transpose graph
- `removeDirections()`          - Turn a directed graph into an undirected graph
- BFS
- `isConnected()`               - check if undirected graph is connected
- `isWeaklyConnected()`         - check if directed graph is weakly connected
- `isStronglyConnected()`       - check if directed graph is strongly connected
- DFS
- Edge Classification
- `isDAG()`                     - Check if the graph is a directed acyclic graph
- Topological Sort
- SCC

## How to use

- Clone the repository:
```bash
git clone https://github.com/YuriMatyash/Algorithms.git
```

- Go over graphAlgorithms.py to see the implementation on each algorithm

- Go over(maybe even run) main.py to see how to use each algorithm

## Project Structure

- `structures/` – Core data structures implementations
- `graphAlgorithms.py` – Algorithm implementations
- `main.py` – Running examples of how to use each algorithm

```plaintext
ALGORITHMS/
│
├── structures/
│   ├── graph.py
│   ├── graphDirected.py
│   ├── graphUndirected.py
│   └── node.py
│
├── graphAlgorithms.py
├── main.py
├── .gitignore
└── README.md
```