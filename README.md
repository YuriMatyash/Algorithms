# Algorithms
A collection of algorithm learnt during Algorithms 1+2 at H.I.T(Holon's Institute of Technology)
All implemented in python for the sake of learning and deep understanding

## Algorithms covered

All of these are covered inside `graphAlgorithms.py`

- BFS
- DFS
- Edge Classification(Using DFS)

## Graph utilities

All of these are covered inside `util.py`

- isDAG()                   - Check if the graph is a directed acyclic graph
- transpose()               - transpose graph
- isConnected()             - check if undirected graph is connected
- removeDirections()        - Turn a directed graph into an undirected graph
- isWeaklyConnected()       - check if directed graph is weakly connected
- isStronglyConnected()     - check if directed graph is strongly connected

## How to use

- Clone the repository:
```bash
git clone https://github.com/YuriMatyash/Algorithms.git
```

- Go over graphAlgorithms.py to see the implementation on each algorithm

- Go over(maybe even run) main.py to see how to use each algorithm

- Go over util.py to see different simple transformations/functions and such done on graphs

## Project Structure

- `structures/` – Core data structures implementations
- `graphAlgorithms.py` – Algorithm implementations
- `main.py` – Running examples of how to use each algorithm
- `util.py` - Different functions or simplistic algorithms for graphs

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
├── README.md
└── util.py
```