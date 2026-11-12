# -*- coding: utf-8 -*-
"""按实验要求一次性运行四个题目的官方测试用例。

用法：
    python main.py
"""

from __future__ import annotations

from .astar import solve_maze, solve_weighted
from .game_tree import build_tree, best_score
from .treasure_game import (KEY_POINTS, best_first_move,
                            compute_distance_matrix, distances_from)

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


def main() -> None:
    print("=" * 72)
    print("题目一：迷宫寻路（A*）")
    print("=" * 72)
    path = solve_maze((0, 0), (3, 4), MAZE)
    expected1 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    print("输出:", path)
    print("预期:", expected1)
    print("校验:", "PASS" if path == expected1 else "FAIL")

    print()
    print("=" * 72)
    print("题目二：加权地形（A*）")
    print("=" * 72)
    cost = solve_weighted((0, 0), (4, 4), WEIGHTED_MAP)
    print("输出:", cost)
    print("预期: 7")
    print("校验:", "PASS" if cost == 7 else "FAIL")

    print()
    print("=" * 72)
    print("题目三：极大极小博弈树（Alpha-Beta）")
    print("=" * 72)
    value = best_score(build_tree(TREE_JSON), "alphabeta")
    print("输出:", int(value) if float(value).is_integer() else value)
    print("预期: 5")
    print("校验:", "PASS" if value == 5 else "FAIL")

    print()
    print("=" * 72)
    print("题目四：竞争环境下的宝藏分配（综合题）")
    print("=" * 72)
    matrix = compute_distance_matrix()
    print("关键点顺序:", KEY_POINTS)
    print("距离矩阵:")
    print("      " + "".join(f"{lab:>6}" for lab in KEY_POINTS))
    for i, lab in enumerate(KEY_POINTS):
        print(f"{lab:>4}  " + "".join(f"{matrix[i][j]:>6}" for j in range(len(KEY_POINTS))))

    dist = distances_from(matrix)
    expected4 = {2: ("T2", -1), 4: ("T3", 3)}
    print()
    for h in (2, 4):
        for alg in ("minimax", "alphabeta"):
            target, value, stats, _ = best_first_move(h, alg, "natural", dist)
            exp_t, exp_v = expected4[h]
            ok = (target, value) == (exp_t, exp_v)
            print(f"H = {h}  [{alg:9s}] => "
                  f'{{"best_target": "{target}", "best_value": {value}}}  '
                  f"(visited={stats.visited}, 剪枝着法={stats.pruned_moves})  "
                  f"校验: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
