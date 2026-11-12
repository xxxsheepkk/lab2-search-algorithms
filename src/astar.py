"""A* 搜索算法实现（实验二 题目一、题目二）。

设计要点
--------
* 状态：网格坐标 ``(row, col)``。
* g(n)：从起点到 n 的实际累计代价。
* h(n)：启发函数，题目一默认曼哈顿距离；题目二提供可采纳性修正版本。
* f(n) = g(n) + h(n)，使用二叉堆（优先队列）按 f 值扩展。
* 图搜索：维护 g_score 字典，允许更优路径时重新入队（无 closed 表硬关闭，
  以避免"启发函数不可采纳 + 关闭表"导致的次优解）。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Coord = Tuple[int, int]
Grid = List[List[str]]

# 上下左右四个方向
DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# 统计信息
# ---------------------------------------------------------------------------
@dataclass
class SearchStats:
    """记录一次 A* 搜索的开销，用于实验对照分析。"""

    expanded: int = 0                 # 被弹出并扩展的节点数
    generated: int = 0                # 生成的后继节点数
    max_frontier: int = 0             # 优先队列峰值大小
    expanded_nodes: List[Coord] = field(default_factory=list)  # 扩展顺序
    cost: Optional[float] = None      # 最终路径代价


# ---------------------------------------------------------------------------
# 启发函数
# ---------------------------------------------------------------------------
def manhattan(a: Coord, b: Coord) -> int:
    """曼哈顿距离，四方向单位代价下可采纳且一致。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero_heuristic(a: Coord, b: Coord) -> int:
    """h = 0，此时 A* 退化为 Dijkstra / 广度优先搜索。"""
    return 0


def default_enter_cost(_cell: str) -> float:
    """题目一的进入代价：任何可通行格代价均为 1。"""
    return 1.0


def weighted_enter_cost(cell: str) -> float:
    """题目二的进入代价：数字地形取数字；S/E 视作 0（终点不计入代价）。"""
    if cell in ("S", "E"):
        return 0.0
    return float(cell)


def in_bounds(grid: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def neighbors(grid: Grid, node: Coord):
    """返回四邻域中可通行的格子（'X' 为障碍）。"""
    r, c = node
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if in_bounds(grid, nr, nc) and grid[nr][nc] != "X":
            yield nr, nc


# ---------------------------------------------------------------------------
# 核心 A*
# ---------------------------------------------------------------------------
def astar(
    start: Coord,
    goal: Coord,
    grid: Grid,
    enter_cost: Callable[[str], float] = default_enter_cost,
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    stats: Optional[SearchStats] = None,
    tie_break: str = "fifo",
) -> Optional[List[Coord]]:
    """在网格上运行 A*。

    参数
    ----
    start, goal : 起止坐标 (row, col)。
    enter_cost  : 进入某格的代价函数，默认每格 1。
    heuristic   : 启发函数 h(n)，默认曼哈顿距离。
    stats       : 可选的统计容器；传入时会记录扩展节点数等信息。
    tie_break   : f 值相同时的次序策略：
                  ``fifo`` 先进先出；``larger_g`` 优先 g 大者（更靠近终点）；
                  ``smaller_h`` 优先 h 小者。后两者是常用工程优化。

    返回
    ----
    从起点到终点的坐标列表（含首尾）；不可达时返回 None。
    起点即终点时返回 ``[start]``。
    """
    if stats is None:
        stats = SearchStats()

    if not in_bounds(grid, *start) or not in_bounds(grid, *goal):
        return None
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return None

    g_score: Dict[Coord, float] = {start: 0.0}
    parent: Dict[Coord, Coord] = {}

    # 堆元素：(f, 次序键, 计数器, g, 节点)
    # 次序键用于同 f 时打破平局；计数器保证 FIFO 且结果可复现。
    counter = 0

    def _priority(f: float, g: float, node: Coord, counter: int):
        if tie_break == "larger_g":
            return (f, -g, counter, g, node)
        if tie_break == "smaller_h":
            return (f, f - g, counter, g, node)
        return (f, 0.0, counter, g, node)

    open_heap: List[Tuple] = []
    heapq.heappush(open_heap, _priority(heuristic(start, goal), 0.0, start, counter))
    stats.max_frontier = 1

    while open_heap:
        f, _, _, g, node = heapq.heappop(open_heap)

        # 惰性删除：堆中旧条目（g 更大）直接跳过
        if g > g_score.get(node, float("inf")):
            continue

        stats.expanded += 1
        stats.expanded_nodes.append(node)

        if node == goal:
            stats.cost = g
            path = [node]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        for nxt in neighbors(grid, node):
            ng = g + enter_cost(grid[nxt[0]][nxt[1]])
            if ng < g_score.get(nxt, float("inf")):
                g_score[nxt] = ng
                parent[nxt] = node
                counter += 1
                heapq.heappush(open_heap, _priority(ng + heuristic(nxt, goal), ng, nxt, counter))
                stats.generated += 1
                stats.max_frontier = max(stats.max_frontier, len(open_heap))

    return None


# ---------------------------------------------------------------------------
# 题目一：迷宫寻路
# ---------------------------------------------------------------------------
def solve_maze(
    start: Coord,
    goal: Coord,
    grid: Grid,
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    stats: Optional[SearchStats] = None,
    tie_break: str = "fifo",
) -> List[Coord]:
    """题目一入口：返回最短路径坐标序列，无解返回 []。"""
    path = astar(start, goal, grid, default_enter_cost, heuristic, stats, tie_break)
    return path if path is not None else []


# ---------------------------------------------------------------------------
# 题目二：加权地形
# ---------------------------------------------------------------------------
def min_step_cost(grid: Grid) -> float:
    """地图中非起终点格子的最小进入代价（用于构造可采纳启发函数）。

    终点 E 的进入代价为 0，不能作为常数下界；S 也不参与移动代价，
    因此统计时排除 S、E、X。
    """
    costs = [weighted_enter_cost(cell) for row in grid for cell in row
             if cell not in ("X", "S", "E")]
    return min(costs) if costs else 1.0


def weighted_heuristic(k: float) -> Callable[[Coord, Coord], float]:
    """返回 h(n) = k * 曼哈顿距离 的启发函数。

    若每个非终点格子进入代价 >= k，则该启发函数可采纳。
    """
    return lambda a, b: k * manhattan(a, b)


def solve_weighted(
    start: Coord,
    goal: Coord,
    grid: Grid,
    heuristic: Optional[Callable[[Coord, Coord], float]] = None,
    stats: Optional[SearchStats] = None,
) -> int:
    """题目二入口：返回最小总代价，无解返回 -1。

    默认启发函数为 ``max(0, 曼哈顿距离 - 1)``：
    由于终点 E 的进入代价为 0，而其余每步至少 1，真实剩余代价 >= d-1，
    因此该启发函数严格可采纳；直接使用曼哈顿距离会在紧邻终点处高估 1。
    """
    if heuristic is None:
        heuristic = lambda a, b: max(0, manhattan(a, b) - 1)
    path = astar(start, goal, grid, weighted_enter_cost, heuristic, stats)
    if path is None:
        return -1
    total = 0.0
    for cell in path[1:]:
        total += weighted_enter_cost(grid[cell[0]][cell[1]])
    return int(total)
