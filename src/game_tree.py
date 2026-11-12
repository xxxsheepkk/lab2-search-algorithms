"""题目三：极大极小博弈树 + Alpha-Beta 剪枝。

树结点既可以用 ``TreeNode`` 手工构造，也可以用 ``build_tree`` 从
JSON 字典（``is_maximizing_player`` / ``value`` / ``children``）构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TreeNode:
    """博弈树结点：叶子有 value，内部结点有 children。"""

    def __init__(self, value: Optional[float] = None,
                 children: Optional[List["TreeNode"]] = None,
                 is_maximizing_player: Optional[bool] = None,
                 name: str = "") -> None:
        self.value = value
        self.children = children if children is not None else []
        self.is_maximizing_player = is_maximizing_player
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        if self.children:
            return f"TreeNode({self.name or 'MAX' if self.is_maximizing_player else 'MIN'}, {len(self.children)} children)"
        return f"TreeNode(value={self.value})"


def build_tree(data: Dict[str, Any], name: str = "root") -> TreeNode:
    """从题目给出的 JSON 结构递归构造树。

    ``is_maximizing_player`` 决定该层是 MAX(True) 还是 MIN(False)；
    叶子结点只含 ``value``。
    """
    if "value" in data and not data.get("children"):
        return TreeNode(value=data["value"], is_maximizing_player=data.get("is_maximizing_player"))
    children = [build_tree(ch, f"{name}.{i}") for i, ch in enumerate(data.get("children", []))]
    return TreeNode(children=children,
                    is_maximizing_player=data.get("is_maximizing_player"),
                    name=name)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
@dataclass
class GameTreeStats:
    """一次博弈树搜索的开销统计。"""

    visited: int = 0          # 访问（进入）的结点数
    leaves: int = 0           # 访问到的叶子数
    pruned: int = 0           # 被剪枝跳过的结点总数（后续兄弟子树）
    cutoffs: int = 0          # 发生剪枝的次数
    trace: List[str] = field(default_factory=list)  # 关键过程记录


# ---------------------------------------------------------------------------
# Minimax（无剪枝）
# ---------------------------------------------------------------------------
def minimax(node: TreeNode, stats: Optional[GameTreeStats] = None) -> float:
    """标准极小极大搜索：MAX 取子结点最大值，MIN 取最小值。"""
    if stats is None:
        stats = GameTreeStats()
    stats.visited += 1

    if not node.children:
        stats.leaves += 1
        return float(node.value)

    # 若结点未显式标注玩家，则按层级推断：根为 MAX，逐层交替
    is_max = node.is_maximizing_player
    if is_max is None:
        is_max = True  # 根节点默认为 MAX（题目约定）

    values = [minimax(ch, stats) for ch in node.children]
    return max(values) if is_max else min(values)


# ---------------------------------------------------------------------------
# Alpha-Beta 剪枝
# ---------------------------------------------------------------------------
def alpha_beta(node: TreeNode, alpha: float = -math.inf, beta: float = math.inf,
               is_maximizing: bool = True, stats: Optional[GameTreeStats] = None,
               order: Optional[List[int]] = None) -> float:
    """Alpha-Beta 剪枝搜索，返回该子树对 MAX 而言的博弈值。

    参数
    ----
    node          : 当前节点。
    alpha, beta   : 当前搜索窗口。
    is_maximizing : 当前节点是否为 MAX 层。
    stats         : 统计容器。
    order         : 子节点访问顺序（下标列表）；None 表示从左到右。
    """
    if stats is None:
        stats = GameTreeStats()
    stats.visited += 1

    if not node.children:
        stats.leaves += 1
        return float(node.value)

    children = node.children if order is None else [node.children[i] for i in order]

    if is_maximizing:
        value = -math.inf
        for idx, ch in enumerate(children):
            value = max(value, alpha_beta(ch, alpha, beta, False, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                # beta 剪枝：MIN 祖先已保证不会选择该分支，
                # 跳过的是后续兄弟子树（而不是刚访问完的 ch）
                stats.cutoffs += 1
                stats.pruned += sum(_count_nodes(s) for s in children[idx + 1:])
                stats.trace.append(f"剪枝 @ {ch.name or 'MAX'} (alpha={alpha}, beta={beta})")
                break
        return value
    else:
        value = math.inf
        for idx, ch in enumerate(children):
            value = min(value, alpha_beta(ch, alpha, beta, True, stats))
            beta = min(beta, value)
            if alpha >= beta:
                # alpha 剪枝，同样跳过后续兄弟子树
                stats.cutoffs += 1
                stats.pruned += sum(_count_nodes(s) for s in children[idx + 1:])
                stats.trace.append(f"剪枝 @ {ch.name or 'MIN'} (alpha={alpha}, beta={beta})")
                break
        return value


def _count_nodes(node: TreeNode) -> int:
    """统计以 node 为根的子树结点数。"""
    return 1 + sum(_count_nodes(ch) for ch in node.children)


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------
def best_score(root: TreeNode, method: str = "alphabeta") -> float:
    """题目三要求：返回 MAX 玩家的最佳结果。"""
    if method == "minimax":
        return minimax(root)
    is_max = root.is_maximizing_player
    return alpha_beta(root, is_maximizing=True if is_max is None else bool(is_max))


def build_example_tree() -> TreeNode:
    """构造 PPT 中的示例树（叶子按从左到右顺序）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2], is_maximizing_player=True)
    max2 = TreeNode(children=[leaf3, leaf4], is_maximizing_player=True)
    max3 = TreeNode(children=[leaf5, leaf6], is_maximizing_player=True)

    min1 = TreeNode(children=[max1, max2], is_maximizing_player=False)
    min2 = TreeNode(children=[max2, max3], is_maximizing_player=False)

    return TreeNode(children=[min1, min2], is_maximizing_player=True, name="root")
