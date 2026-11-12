# -*- coding: utf-8 -*-
"""实验二测试用例：覆盖四个题目的官方样例与边界情况。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.astar import SearchStats, manhattan, solve_maze, solve_weighted, zero_heuristic
from src.game_tree import (alpha_beta, best_score, build_tree,
                           build_example_tree, minimax)
from src.treasure_game import (KEY_POINTS, best_first_move,
                               compute_distance_matrix, distances_from, initial_state)


# ---------------------------------------------------------------------------
# 官方地图
# ---------------------------------------------------------------------------
MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]

WEIGHTED_MAP = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]

TREE_JSON = {
    "tree_structure": {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                ],
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                    {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
                ],
            },
        ],
    }
}


class TestProblem1Maze(unittest.TestCase):
    def test_official_case(self):
        expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
        self.assertEqual(solve_maze((0, 0), (3, 4), MAZE), expected)

    def test_path_length_is_optimal(self):
        path = solve_maze((0, 0), (3, 4), MAZE)
        self.assertEqual(len(path) - 1, 7)  # 最短 7 步

    def test_unreachable_returns_empty(self):
        grid = [["S", "X", "E"], [".", "X", "."], [".", "X", "."]]
        self.assertEqual(solve_maze((0, 0), (0, 2), grid), [])

    def test_start_equals_goal(self):
        self.assertEqual(solve_maze((0, 0), (0, 0), MAZE), [(0, 0)])

    def test_h_zero_matches_manhattan_cost(self):
        stats_m, stats_d = SearchStats(), SearchStats()
        p1 = solve_maze((0, 0), (3, 4), MAZE, heuristic=manhattan, stats=stats_m)
        p2 = solve_maze((0, 0), (3, 4), MAZE, heuristic=zero_heuristic, stats=stats_d)
        self.assertEqual(len(p1), len(p2))
        self.assertGreaterEqual(stats_d.expanded, stats_m.expanded)


class TestProblem2Weighted(unittest.TestCase):
    def test_official_case(self):
        self.assertEqual(solve_weighted((0, 0), (4, 4), WEIGHTED_MAP), 7)

    def test_unreachable_returns_minus_one(self):
        grid = [["S", "X", "E"], ["1", "X", "1"], ["1", "X", "1"]]
        self.assertEqual(solve_weighted((0, 0), (0, 2), grid), -1)

    def test_e_cell_cost_is_zero(self):
        # 单行走廊：进入两个 1 格，E 不计代价
        grid = [["S", "1", "1", "E"]]
        self.assertEqual(solve_weighted((0, 0), (0, 3), grid), 2)

    def test_all_heuristics_agree_on_cost(self):
        for h in (zero_heuristic,
                  manhattan,
                  lambda a, b: max(0, manhattan(a, b) - 1)):
            self.assertEqual(solve_weighted((0, 0), (4, 4), WEIGHTED_MAP, heuristic=h), 7)


class TestProblem3GameTree(unittest.TestCase):
    def test_official_case_alphabeta(self):
        self.assertEqual(best_score(build_example_tree(), "alphabeta"), 5)

    def test_official_case_minimax(self):
        self.assertEqual(best_score(build_example_tree(), "minimax"), 5)

    def test_json_build(self):
        self.assertEqual(best_score(build_tree(TREE_JSON["tree_structure"]), "alphabeta"), 5)

    def test_root_defaults_to_max_when_flag_missing(self):
        root = build_example_tree()
        root.is_maximizing_player = None
        self.assertEqual(best_score(root, "alphabeta"), 5)

    def test_alphabeta_equals_minimax_random_trees(self):
        rng = random.Random(20261112)

        def make_tree(depth, is_max):
            if depth == 0:
                return {"value": rng.randint(-20, 20)}
            n = rng.randint(2, 4)
            return {
                "is_maximizing_player": is_max,
                "children": [make_tree(depth - 1, not is_max) for _ in range(n)],
            }

        for _ in range(200):
            root = build_tree(make_tree(3, True))
            self.assertEqual(minimax(root), alpha_beta(root, is_maximizing=True))


class TestProblem4Treasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = compute_distance_matrix()
        cls.dist = distances_from(cls.matrix)

    def test_distance_matrix_symmetric(self):
        for i in range(len(KEY_POINTS)):
            self.assertEqual(self.matrix[i][i], 0)
            for j in range(len(KEY_POINTS)):
                self.assertEqual(self.matrix[i][j], self.matrix[j][i])

    def test_distance_matrix_values(self):
        expected = {
            ("A", "B"): 12, ("A", "T1"): 10, ("A", "T2"): 4, ("A", "T3"): 8, ("A", "T4"): 8,
            ("B", "T1"): 8, ("B", "T2"): 8, ("B", "T3"): 4, ("B", "T4"): 10,
            ("T1", "T2"): 6, ("T1", "T3"): 6, ("T1", "T4"): 12,
            ("T2", "T3"): 6, ("T2", "T4"): 6, ("T3", "T4"): 6,
        }
        for (a, b), v in expected.items():
            self.assertEqual(self.dist[(a, b)], v)

    def test_initial_state(self):
        s = initial_state()
        self.assertEqual((s.pos["A"], s.pos["B"]), ("A", "B"))
        self.assertEqual((s.score["A"], s.score["B"]), (0, 0))
        self.assertEqual(s.remaining, ("T1", "T2", "T3", "T4"))
        self.assertEqual(s.value(), 0)

    def test_h2_expected(self):
        for alg in ("minimax", "alphabeta"):
            target, value, _, _ = best_first_move(2, alg, "natural", self.dist)
            self.assertEqual((target, value), ("T2", -1))

    def test_h4_expected(self):
        for alg in ("minimax", "alphabeta"):
            target, value, _, _ = best_first_move(4, alg, "natural", self.dist)
            self.assertEqual((target, value), ("T3", 3))

    def test_alphabeta_equals_minimax_all_depths_orders(self):
        for depth in (1, 2, 3, 4):
            base = best_first_move(depth, "minimax", "natural", self.dist)[1]
            for order in ("natural", "value_desc", "gain_desc", "gain_asc"):
                self.assertEqual(best_first_move(depth, "alphabeta", order, self.dist)[1], base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
