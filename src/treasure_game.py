"""题目四：竞争环境下的宝藏分配（A* + Minimax / Alpha-Beta 综合题）。

建模
----
* 位置集合：A=(0,0), B=(6,6), T1..T4，共 6 个关键点；
  两两之间用加权 A* 求最小移动代价，得到 6x6 距离矩阵。
* 博弈状态：``GameState(pos_a, pos_b, remaining, score_a, score_b, turn)``，
  ``remaining`` 为尚未被获取的宝藏集合，``turn`` 为当前行动方。
* 每回合选一个剩余宝藏 t：玩家移动到 t，获得 ``V(t) - d(pos, t)`` 收益并更新位置。
* 终局/深度截止时评价值 U = score_a - score_b；A 最大化 U，B 最小化 U。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .astar import Coord, SearchStats, astar, weighted_enter_cost

# ---------------------------------------------------------------------------
# 地图与宝藏（来自实验要求第 14 页）
# ---------------------------------------------------------------------------
# 'X' 为障碍；数字为进入代价；A/B/T 所在格按规则代价为 1（写作 'A'/'B'/'T'）
MAP: List[List[str]] = [
    ["A", "1", "1", "X", "2", "1", "T"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T", "1", "1", "X", "1", "1", "B"],
]

POSITIONS: Dict[str, Coord] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}

TREASURE_VALUES: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}

KEY_POINTS: Tuple[str, ...] = ("A", "B", "T1", "T2", "T3", "T4")
TREASURES: Tuple[str, ...] = ("T1", "T2", "T3", "T4")


def game_enter_cost(cell: str) -> float:
    """题目四的进入代价：A/B/T 格为 1，数字取数字，X 不可通行。"""
    if cell in ("A", "B", "T", "S", "E"):
        return 1.0
    return float(cell)


# ---------------------------------------------------------------------------
# 距离矩阵
# ---------------------------------------------------------------------------
def compute_distance_matrix(verbose: bool = False) -> List[List[int]]:
    """计算 A、B、T1..T4 两两之间的最小移动代价（6x6，对称）。"""
    n = len(KEY_POINTS)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p, q = POSITIONS[KEY_POINTS[i]], POSITIONS[KEY_POINTS[j]]
            path = astar(p, q, MAP, game_enter_cost, lambda a, b: 0)
            if path is None:
                raise ValueError(f"{KEY_POINTS[i]} 与 {KEY_POINTS[j]} 不连通")
            cost = sum(game_enter_cost(MAP[r][c]) for r, c in path[1:])
            matrix[i][j] = matrix[j][i] = int(cost)
    if verbose:
        print_distance_matrix(matrix)
    return matrix


def print_distance_matrix(matrix: Sequence[Sequence[int]]) -> None:
    labels = KEY_POINTS
    print("距离矩阵 d(p, q)：")
    print("      " + "".join(f"{lab:>6}" for lab in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:>4}  " + "".join(f"{matrix[i][j]:>6}" for j in range(len(labels))))


def distances_from(matrix: Sequence[Sequence[int]]) -> Dict[Tuple[str, str], int]:
    return {(KEY_POINTS[i], KEY_POINTS[j]): matrix[i][j]
            for i in range(len(KEY_POINTS)) for j in range(len(KEY_POINTS))}


# ---------------------------------------------------------------------------
# 博弈状态
# ---------------------------------------------------------------------------
Player = str  # 'A' 或 'B'


@dataclass(frozen=True)
class GameState:
    """博弈状态：双方位置、剩余宝藏、累计收益、当前玩家。"""

    pos: Dict[Player, str]          # {'A': 'A', 'B': 'B'}，值为关键点名称
    remaining: Tuple[str, ...]
    score: Dict[Player, int]        # {'A': 累计收益, 'B': 累计收益}
    turn: Player

    def value(self) -> int:
        """U = Score_A - Score_B"""
        return self.score["A"] - self.score["B"]

    def describe(self) -> str:
        return (f"turn={self.turn}, posA={self.pos['A']}, posB={self.pos['B']}, "
                f"rem={list(self.remaining)}, scoreA={self.score['A']}, "
                f"scoreB={self.score['B']}, U={self.value()}")


def initial_state() -> GameState:
    return GameState(pos={"A": "A", "B": "B"},
                     remaining=TREASURES,
                     score={"A": 0, "B": 0},
                     turn="A")


# ---------------------------------------------------------------------------
# 搜索统计
# ---------------------------------------------------------------------------
@dataclass
class GameStats:
    visited: int = 0        # 展开的状态数
    pruned_moves: int = 0   # 因剪枝而不再展开的着法数
    cutoffs: int = 0        # 剪枝次数
    evaluations: int = 0    # 叶子评估次数


def _ordered_moves(state: GameState, dist: Dict[Tuple[str, str], int],
                   order: str) -> List[str]:
    """生成候选宝藏并排序。order ∈ {natural, value_desc, gain_desc}。"""
    moves = list(state.remaining)
    if order == "natural":
        return moves
    if order == "value_desc":
        return sorted(moves, key=lambda t: -TREASURE_VALUES[t])
    if order == "gain_desc":
        return sorted(moves, key=lambda t: -(TREASURE_VALUES[t] - dist[(state.pos[state.turn], t)]))
    if order == "gain_asc":
        return sorted(moves, key=lambda t: TREASURE_VALUES[t] - dist[(state.pos[state.turn], t)])
    raise ValueError(f"未知排序方式: {order}")


def _apply(state: GameState, target: str,
           dist: Dict[Tuple[str, str], int]) -> Tuple[GameState, int]:
    """执行一步：当前玩家移动到 target、获得收益、切换行动方。"""
    player = state.turn
    gain = TREASURE_VALUES[target] - dist[(state.pos[player], target)]
    new_pos = dict(state.pos)
    new_pos[player] = target
    new_score = dict(state.score)
    new_score[player] += gain
    next_turn = "B" if player == "A" else "A"
    return GameState(pos=new_pos,
                     remaining=tuple(t for t in state.remaining if t != target),
                     score=new_score,
                     turn=next_turn), gain


# ---------------------------------------------------------------------------
# Minimax（无剪枝）
# ---------------------------------------------------------------------------
def minimax(state: GameState, depth: int, dist: Dict[Tuple[str, str], int],
            order: str = "natural", stats: Optional[GameStats] = None) -> int:
    """返回当前状态在给定深度下的博弈值。"""
    if stats is None:
        stats = GameStats()
    stats.visited += 1

    if depth == 0 or not state.remaining:
        stats.evaluations += 1
        return state.value()

    moves = _ordered_moves(state, dist, order)
    if state.turn == "A":
        best = -math.inf
        for t in moves:
            child, _ = _apply(state, t, dist)
            best = max(best, minimax(child, depth - 1, dist, order, stats))
        return int(best)
    else:
        best = math.inf
        for t in moves:
            child, _ = _apply(state, t, dist)
            best = min(best, minimax(child, depth - 1, dist, order, stats))
        return int(best)


# ---------------------------------------------------------------------------
# Alpha-Beta
# ---------------------------------------------------------------------------
def alpha_beta(state: GameState, depth: int, dist: Dict[Tuple[str, str], int],
               alpha: float = -math.inf, beta: float = math.inf,
               order: str = "natural", stats: Optional[GameStats] = None) -> int:
    """Alpha-Beta 剪枝版 Minimax，返回博弈值。"""
    if stats is None:
        stats = GameStats()
    stats.visited += 1

    if depth == 0 or not state.remaining:
        stats.evaluations += 1
        return state.value()

    moves = _ordered_moves(state, dist, order)
    if state.turn == "A":
        value = -math.inf
        for idx, t in enumerate(moves):
            child, _ = _apply(state, t, dist)
            value = max(value, alpha_beta(child, depth - 1, dist, alpha, beta, order, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.cutoffs += 1
                stats.pruned_moves += len(moves) - idx - 1
                break
        return int(value)
    else:
        value = math.inf
        for idx, t in enumerate(moves):
            child, _ = _apply(state, t, dist)
            value = min(value, alpha_beta(child, depth - 1, dist, alpha, beta, order, stats))
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                stats.pruned_moves += len(moves) - idx - 1
                break
        return int(value)


# ---------------------------------------------------------------------------
# 求最优首步
# ---------------------------------------------------------------------------
def best_first_move(depth: int, algorithm: str = "alphabeta",
                    order: str = "natural",
                    dist: Optional[Dict[Tuple[str, str], int]] = None):
    """求 A 的最优首步与对应博弈值。

    返回 ``(best_target, best_value, stats, ranked)``，
    ``ranked`` 为每个首步的 ``(target, value)`` 列表。
    """
    if dist is None:
        dist = distances_from(compute_distance_matrix())
    root = initial_state()
    stats = GameStats()
    search = alpha_beta if algorithm == "alphabeta" else minimax

    ranked = []
    for t in root.remaining:
        child, gain = _apply(root, t, dist)
        value = search(child, depth - 1, dist, order=order, stats=stats)

        # 根层记录收益便于报告分析
        ranked.append({"target": t, "gain": gain, "value": value})

    best = max(ranked, key=lambda x: x["value"])
    return best["target"], best["value"], stats, ranked


def principal_variation(depth: int, dist: Dict[Tuple[str, str], int],
                        order: str = "natural") -> List[Tuple[str, str, int, int]]:
    """返回 Alpha-Beta 意义下的主变例。

    每项为 ``(玩家, 选择的宝藏, 本步收益, 该步之后的 U)``。
    """
    state = initial_state()
    line: List[Tuple[str, str, int, int]] = []
    remaining_depth = depth
    while remaining_depth > 0 and state.remaining:
        if state.turn == "A":
            best_t, best_val = None, -math.inf
            for t in _ordered_moves(state, dist, order):
                child, _ = _apply(state, t, dist)
                v = alpha_beta(child, remaining_depth - 1, dist, order=order)
                if v > best_val:
                    best_t, best_val = t, v
        else:
            best_t, best_val = None, math.inf
            for t in _ordered_moves(state, dist, order):
                child, _ = _apply(state, t, dist)
                v = alpha_beta(child, remaining_depth - 1, dist, order=order)
                if v < best_val:
                    best_t, best_val = t, v
        child, gain = _apply(state, best_t, dist)
        line.append((state.turn, best_t, gain, child.value()))
        state = child
        remaining_depth -= 1
    return line
