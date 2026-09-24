"""
Binary SumTree Data Structure for Prioritized Experience Replay (PER).
Provides O(log N) priority updates and O(log N) prefix-sum sampling.
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Any
import numpy as np


class SumTree:
    """
    Binary Sum Tree where each parent node holds the sum of its children.
    Leaf nodes store individual transition priorities.
    Internal nodes store the intermediate cumulative sums.

    Array indexing:
      - Internal nodes: indices 0 to capacity - 2
      - Leaf nodes: indices capacity - 1 to 2 * capacity - 2
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        # 2 * capacity - 1 total nodes in a complete binary tree with capacity leaves
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        # Array to store actual transition payloads
        self.data = np.zeros(self.capacity, dtype=object)
        self.pointer = 0
        self.size = 0

    def add(self, priority: float, experience: Any) -> None:
        """Add experience with priority to the next leaf position (FIFO overwrite)."""
        tree_idx = self.pointer + self.capacity - 1
        self.data[self.pointer] = experience
        self.update(tree_idx, priority)

        self.pointer = (self.pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float) -> None:
        """Update leaf priority and propagate change upward to root in O(log N)."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate upward to parent nodes
        curr_idx = tree_idx
        while curr_idx != 0:
            curr_idx = (curr_idx - 1) // 2
            self.tree[curr_idx] += delta

    def get_leaf(self, value: float) -> Tuple[int, float, Any]:
        """
        Traverse down tree to find the leaf corresponding to the cumulative sum value.
        Runs in O(log N) time.
        """
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1

            # Reached beyond tree bounds -> parent is the leaf
            if left_child >= len(self.tree):
                leaf_idx = parent
                break

            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Cumulative priority sum across all leaves stored at root."""
        return float(self.tree[0])

    def __len__(self) -> int:
        return self.size
