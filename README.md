# Graph Algorithms and Data Structures

This project implements a variety of graph-related algorithms and data structures, including vertex and graph classes, priority queues, and search algorithms like BFS and A*.

## Overview

The project provides an implementation of:

- **Graph and Vertex Classes:** Represent graph structures using adjacency maps.
- **Priority Queues:** Custom priority queues built on Python's `heapq` module, used for graph traversal algorithms.
- **Graph Algorithms:** Includes BFS, A* Search, and Tollway Algorithm for shortest path finding with coupons.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/vanshikadian/Vertex_Object.git
```

## Usage
Import the classes and functions into your project to utilize the graph structures and algorithms. No additional libraries are needed beyond the Python standard library and numpy for matrix handling.

## Classes
### Vertex
Represents a vertex in a graph with properties such as ID, adjacency list, coordinates, and visited status.
Methods include computing Euclidean and taxicab distances to other vertices.
### Graph
Implements a graph using an adjacency map structure.
Supports operations like adding vertices and edges, converting between matrix representations, and performing search algorithms.
### PriorityQueue and TollWayPriorityQueue
Custom priority queues with support for priority updates, used in graph algorithms.

## Algorithms
### Breadth-First Search (BFS)
Finds the shortest path between two vertices in an unweighted graph.

### A* Search
Finds the shortest path using a heuristic function to estimate the distance to the target vertex.

### Tollway Algorithm
Finds the cheapest path in a graph where tolls can be reduced by using a limited number of coupons.

