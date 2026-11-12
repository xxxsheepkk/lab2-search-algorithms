# -*- coding: utf-8 -*-
"""实验二对照实验：生成报告所需的全部数据、表格与图表。

运行：
    python experiments/run_experiments.py

输出：
    results/*.csv        原始实验数据
    results/summary.json 关键结论汇总
    report/figures/*.svg 图表
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart import grouped_bar_chart, heatmap, line_chart
from src.astar import (SearchStats, astar, manhattan, solve_maze, solve_weighted,
                       weighted_enter_cost, zero_heuristic)
from src.game_tree import (GameTreeStats, TreeNode, alpha_beta, build_tree,
                           minimax)
from src.run_all import TREE_JSON, MAZE as MAZE_OFFICIAL, WEIGHTED_MAP
from src.treasure_game import (KEY_POINTS, MAP as TREASURE_MAP, TREASURES,
                               best_first_move, compute_distance_matrix,
                               distances_from, principal_variation)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "report", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

SIZE = 18          # 随机网格边长
N_MAZES = 40       # 每个障碍密度下的随机迷宫数
N_WEIGHTED = 40    # 随机加权地图数
SEED = 20261112


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def write_csv(name: str, header: Sequence[str], rows: Sequence[Sequence]) -> str:
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def random_maze(n: int, density: float, rng: random.Random) -> List[List[str]]:
    grid = [["." if rng.random() >= density else "X" for _ in range(n)] for _ in range(n)]
    grid[0][0] = "S"
    grid[n - 1][n - 1] = "E"
    return grid


def random_weighted_map(n: int, density: float, rng: random.Random) -> List[List[str]]:
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append("X" if rng.random() < density else str(rng.randint(1, 9)))
        grid.append(row)
    grid[0][0] = "S"
    grid[n - 1][n - 1] = "E"
    return grid


def count_path_cost(path: Sequence[Tuple[int, int]], grid, cost_fn) -> int:
    return int(sum(cost_fn(grid[r][c]) for r, c in path[1:]))


# ---------------------------------------------------------------------------
# 实验一：曼哈顿启发函数 vs h=0；平局策略
# ---------------------------------------------------------------------------
def experiment_problem1() -> Dict:
    print("[1/5] 题目一：h=0 / 曼哈顿 / 平局策略对照 ...")
    # 官方样例
    official = []
    for name, h, tie in [("A*(曼哈顿)", manhattan, "fifo"),
                         ("A*(曼哈顿, larger_g)", manhattan, "larger_g"),
                         ("Dijkstra(h=0)", zero_heuristic, "fifo")]:
        stats = SearchStats()
        t0 = time.perf_counter()
        path = solve_maze((0, 0), (3, 4), MAZE_OFFICIAL, h, stats, tie)
        elapsed = (time.perf_counter() - t0) * 1000
        official.append({
            "method": name, "expanded": stats.expanded, "generated": stats.generated,
            "max_frontier": stats.max_frontier, "path_len": len(path) - 1,
            "cost": stats.cost, "time_ms": round(elapsed, 4), "path": path,
        })

    # 随机迷宫：不同障碍密度
    densities = [0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    rng = random.Random(SEED)
    for density in densities:
        acc = {"h0": [], "manhattan": [], "tie": [], "length": []}
        used = 0
        attempts = 0
        while used < N_MAZES and attempts < N_MAZES * 20:
            attempts += 1
            grid = random_maze(SIZE, density, rng)
            base = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid, zero_heuristic)
            if not base:
                continue
            s0, s1, s2 = SearchStats(), SearchStats(), SearchStats()
            p0 = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid, zero_heuristic, s0)
            p1 = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid, manhattan, s1)
            p2 = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid, manhattan, s2, "larger_g")
            assert len(p0) == len(p1) == len(p2), "A* 与 Dijkstra 路径长度不一致"
            acc["h0"].append(s0.expanded)
            acc["manhattan"].append(s1.expanded)
            acc["tie"].append(s2.expanded)
            acc["length"].append(len(p0) - 1)
            used += 1
        rows.append([density, used,
                     round(statistics.mean(acc["h0"]), 2),
                     round(statistics.mean(acc["manhattan"]), 2),
                     round(statistics.mean(acc["tie"]), 2),
                     round(statistics.mean(acc["length"]), 2)])

    write_csv("p1_maze_vs_density.csv",
              ["obstacle_density", "n_mazes", "avg_expanded_h0",
               "avg_expanded_manhattan", "avg_expanded_tie_larger_g", "avg_path_len"],
              rows)

    grouped_bar_chart(
        os.path.join(FIGURES, "fig1_p1_expanded.svg"),
        "题目一：不同障碍密度下的平均扩展节点数（18×18，40 个随机迷宫）",
        [f"{d:.2f}" for d in densities],
        [("Dijkstra (h=0)", [r[2] for r in rows], "#8a8a8a"),
         ("A* 曼哈顿", [r[3] for r in rows], "#3b6ea5"),
         ("A* 曼哈顿 + larger-g", [r[4] for r in rows], "#4c9a6a")],
        ylabel="平均扩展节点数", value_fmt="{:.0f}")

    return {"official": official, "density_rows": rows}


# ---------------------------------------------------------------------------
# 拓展实验：加权 A* 的扩展数与路径长度权衡
# ---------------------------------------------------------------------------
def experiment_weighted_astar() -> Dict:
    print("[1b/5] 拓展：加权 A* 的速度—最优性权衡 ...")
    rng = random.Random(SEED + 7)
    grids = []
    attempts = 0
    while len(grids) < N_MAZES and attempts < N_MAZES * 20:
        attempts += 1
        grid = random_maze(SIZE, 0.20, rng)
        base = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid, zero_heuristic)
        if base:
            grids.append((grid, len(base) - 1))

    weights = [1.0, 1.5, 2.0, 2.5, 3.0]
    rows = []
    base_exp = None
    for w in weights:
        exps, lens = [], []
        for grid, _opt_len in grids:
            s = SearchStats()
            path = solve_maze((0, 0), (SIZE - 1, SIZE - 1), grid,
                              lambda a, b, w=w: w * manhattan(a, b), s)
            exps.append(s.expanded)
            lens.append(len(path) - 1)
        if base_exp is None:
            base_exp = statistics.mean(exps)
        subopt = sum(1 for (_, opt), ln in zip(grids, lens) if ln != opt)
        ratios = [ln / opt for (_, opt), ln in zip(grids, lens)]
        rows.append([w, round(statistics.mean(exps), 2), round(statistics.mean(lens), 2),
                     round(statistics.mean(ratios), 4), round(max(ratios), 4), subopt,
                     len(grids)])

    header = ["w", "avg_expanded", "avg_path_len", "avg_path_ratio",
              "max_path_ratio", "suboptimal_maps", "n_maps"]
    write_csv("p1_weighted_astar.csv", header, rows)

    grouped_bar_chart(
        os.path.join(FIGURES, "fig8_p1_weighted_astar.svg"),
        "题目一拓展：加权 A* 的扩展数与路径长度权衡（40 个 18×18 迷宫）",
        [f"w={w:g}" for w in weights],
        [("扩展节点数（相对 w=1）", [r[1] / base_exp * 100 for r in rows], "#3b6ea5"),
         ("路径长度（相对最优）", [r[3] * 100 for r in rows], "#e08a3c")],
        ylabel="相对百分比（%）", value_fmt="{:.1f}", width=820)

    return {"rows": rows, "header": header}


# ---------------------------------------------------------------------------
# 实验二：加权地形与启发函数可采纳性
# ---------------------------------------------------------------------------
def experiment_problem2() -> Dict:
    print("[2/5] 题目二：加权地形 / 启发函数可采纳性 ...")
    start, goal = (0, 0), (4, 4)
    # 最优路径逐步代价
    stats = SearchStats()
    path = astar(start, goal, WEIGHTED_MAP, weighted_enter_cost, manhattan, stats)
    assert path is not None
    accum, g = [], 0
    for i, (r, c) in enumerate(path):
        cell = WEIGHTED_MAP[r][c]
        step_cost = 0 if i == 0 else weighted_enter_cost(cell)
        g += step_cost
        accum.append([i, f"({r},{c})", cell, step_cost, g,
                      manhattan((r, c), goal), g + manhattan((r, c), goal)])
    write_csv("p2_path_accumulation.csv",
              ["step", "coord", "cell", "enter_cost", "g", "h_manhattan", "f"],
              accum)

    # 启发函数对照（官方地图）
    official = []
    for name, h in [("Dijkstra (h=0)", zero_heuristic),
                    ("曼哈顿距离", manhattan),
                    ("修正曼哈顿 max(0,d-1)", lambda a, b: max(0, manhattan(a, b) - 1))]:
        s = SearchStats()
        t0 = time.perf_counter()
        cost = solve_weighted(start, goal, WEIGHTED_MAP, h, s)
        official.append({
            "method": name, "cost": cost, "expanded": s.expanded,
            "generated": s.generated, "max_frontier": s.max_frontier,
            "time_ms": round((time.perf_counter() - t0) * 1000, 4),
        })
    write_csv("p2_official_heuristics.csv",
              ["method", "cost", "expanded", "generated", "max_frontier", "time_ms"],
              [[o["method"], o["cost"], o["expanded"], o["generated"],
                o["max_frontier"], o["time_ms"]] for o in official])

    # 随机加权地图：比较扩展节点与正确性
    rng = random.Random(SEED + 2)
    rows, mismatch = [], 0
    used = 0
    while used < N_WEIGHTED:
        grid = random_weighted_map(SIZE, 0.20, rng)
        base_stats = SearchStats()
        base = solve_weighted((0, 0), (SIZE - 1, SIZE - 1), grid, zero_heuristic, base_stats)
        if base == -1:
            continue
        s_m, s_c = SearchStats(), SearchStats()
        c_m = solve_weighted((0, 0), (SIZE - 1, SIZE - 1), grid, manhattan, s_m)
        c_c = solve_weighted((0, 0), (SIZE - 1, SIZE - 1), grid,
                             lambda a, b: max(0, manhattan(a, b) - 1), s_c)
        if c_m != base:
            mismatch += 1
        rows.append([used, base, c_m, c_c, base_stats.expanded,
                     s_m.expanded, s_c.expanded])
        used += 1

    summary = {
        "n_maps": len(rows),
        "cost_mismatch_manhattan": mismatch,
        "avg_expanded_h0": round(statistics.mean(r[4] for r in rows), 2),
        "avg_expanded_manhattan": round(statistics.mean(r[5] for r in rows), 2),
        "avg_expanded_corrected": round(statistics.mean(r[6] for r in rows), 2),
        "avg_speedup_manhattan": round(statistics.mean(r[4] / max(1, r[5]) for r in rows), 3),
        "avg_speedup_corrected": round(statistics.mean(r[4] / max(1, r[6]) for r in rows), 3),
    }
    write_csv("p2_random_heuristics.csv",
              ["map_id", "cost_h0", "cost_manhattan", "cost_corrected",
               "expanded_h0", "expanded_manhattan", "expanded_corrected"], rows)

    grouped_bar_chart(
        os.path.join(FIGURES, "fig2_p2_heuristics.svg"),
        "题目二：随机加权地图上的平均扩展节点数（18×18，40 张地图）",
        ["Dijkstra (h=0)", "曼哈顿", "修正曼哈顿"],
        [("平均扩展节点数", [summary["avg_expanded_h0"], summary["avg_expanded_manhattan"],
                       summary["avg_expanded_corrected"]], "#3b6ea5")],
        ylabel="平均扩展节点数", value_fmt="{:.1f}", width=700)

    return {"official": official, "path": accum, "summary": summary}


# ---------------------------------------------------------------------------
# 实验三：Minimax vs Alpha-Beta 与子节点顺序
# ---------------------------------------------------------------------------
def clone_with_order(node: TreeNode, mode: str,
                     rng: random.Random | None = None) -> TreeNode:
    """按指定策略重排每个内部节点的子节点。"""
    if not node.children:
        return TreeNode(value=node.value, is_maximizing_player=node.is_maximizing_player)
    children = [clone_with_order(ch, mode, rng) for ch in node.children]
    if mode == "best":
        # 完美着法排序：MAX 先走分值最大的分支，MIN 先走分值最小的分支
        children.sort(key=minimax, reverse=bool(node.is_maximizing_player))
    elif mode == "worst":
        children.sort(key=minimax, reverse=not bool(node.is_maximizing_player))
    elif mode == "random" and rng is not None:
        rng.shuffle(children)
    elif mode not in ("natural", "random"):
        raise ValueError(mode)
    return TreeNode(children=children, is_maximizing_player=node.is_maximizing_player,
                    name=node.name)


def experiment_problem3() -> Dict:
    print("[3/5] 题目三：Alpha-Beta 与访问顺序 ...")
    tree_data = TREE_JSON.get("tree_structure", TREE_JSON)
    root = build_tree(tree_data)

    mm_stats = GameTreeStats()
    mm_value = minimax(root, mm_stats)

    variants = []
    for mode in ("natural", "best", "worst"):
        stats = GameTreeStats()
        value = alpha_beta(clone_with_order(root, mode), stats=stats)
        variants.append({"order": mode, "value": value,
                         "visited": stats.visited, "leaves": stats.leaves,
                         "cutoffs": stats.cutoffs, "pruned": stats.pruned})
    # 随机顺序取多次平均
    rng = random.Random(SEED + 3)
    rand_visited = []
    for _ in range(200):
        stats = GameTreeStats()
        alpha_beta(clone_with_order(root, "random", rng), stats=stats)
        rand_visited.append(stats.visited)
    variants.append({"order": "random(avg200)", "value": mm_value,
                     "visited": round(statistics.mean(rand_visited), 2),
                     "leaves": "-", "cutoffs": "-", "pruned": "-"})

    write_csv("p3_official_tree.csv",
              ["method", "value", "visited", "leaves", "cutoffs", "pruned_nodes"],
              [["Minimax", mm_value, mm_stats.visited, mm_stats.leaves, 0, 0]] +
              [[f"Alpha-Beta({v['order']})", v["value"], v["visited"], v["leaves"],
                v["cutoffs"], v["pruned"]] for v in variants])

    grouped_bar_chart(
        os.path.join(FIGURES, "fig3_p3_official.svg"),
        "题目三：示例树在不同访问顺序下的结点访问数",
        ["Minimax", "Alpha-Beta\n原始顺序", "Alpha-Beta\n最佳顺序", "Alpha-Beta\n最差顺序"],
        [("访问结点数", [mm_stats.visited] + [v["visited"] for v in variants[:3]], "#3b6ea5")],
        ylabel="访问结点数", value_fmt="{:.0f}", width=720)

    # 随机博弈树：深度 4，分支 2~3
    rng = random.Random(SEED + 4)
    rows = []
    n_trees = 60
    for i in range(n_trees):
        data = random_tree_dict(4, True, rng, (2, 3))
        root_i = build_tree(data)
        s_mm = GameTreeStats()
        minimax(root_i, s_mm)
        row = [i, s_mm.visited]
        for mode in ("natural", "best", "worst"):
            s = GameTreeStats()
            alpha_beta(clone_with_order(root_i, mode), stats=s)
            row.extend([s.visited, s.pruned])
        rand_visits = []
        for _ in range(10):
            s = GameTreeStats()
            alpha_beta(clone_with_order(root_i, "random", rng), stats=s)
            rand_visits.append(s.visited)
        row.append(round(statistics.mean(rand_visits), 2))
        rows.append(row)

    write_csv("p3_random_trees.csv",
              ["tree_id", "minimax_visited", "ab_natural_visited", "ab_natural_pruned",
               "ab_best_visited", "ab_best_pruned", "ab_worst_visited",
               "ab_worst_pruned", "ab_random_visited"], rows)

    summary = {
        "n_trees": n_trees,
        "avg_minimax_visited": round(statistics.mean(r[1] for r in rows), 2),
        "avg_ab_natural": round(statistics.mean(r[2] for r in rows), 2),
        "avg_ab_best": round(statistics.mean(r[4] for r in rows), 2),
        "avg_ab_worst": round(statistics.mean(r[6] for r in rows), 2),
        "avg_ab_random": round(statistics.mean(r[8] for r in rows), 2),
    }
    summary["reduction_natural_pct"] = round(
        100 * (1 - summary["avg_ab_natural"] / summary["avg_minimax_visited"]), 2)
    summary["reduction_best_pct"] = round(
        100 * (1 - summary["avg_ab_best"] / summary["avg_minimax_visited"]), 2)

    grouped_bar_chart(
        os.path.join(FIGURES, "fig4_p3_random.svg"),
        "题目三：随机博弈树（深度 4，分支 2~3，60 棵）平均访问结点数",
        ["Minimax", "AB-原始", "AB-最佳", "AB-最差", "AB-随机"],
        [("平均访问结点数", [summary["avg_minimax_visited"], summary["avg_ab_natural"],
                       summary["avg_ab_best"], summary["avg_ab_worst"],
                       summary["avg_ab_random"]], "#3b6ea5")],
        ylabel="平均访问结点数", value_fmt="{:.1f}", width=760)

    return {"official_value": mm_value, "variants": variants, "summary": summary}


def random_tree_dict(depth: int, is_max: bool, rng: random.Random,
                     branch: Tuple[int, int]) -> Dict:
    if depth == 0:
        return {"value": rng.randint(-50, 50)}
    n = rng.randint(*branch)
    return {"is_maximizing_player": is_max,
            "children": [random_tree_dict(depth - 1, not is_max, rng, branch)
                         for _ in range(n)]}


# ---------------------------------------------------------------------------
# 实验四：宝藏分配
# ---------------------------------------------------------------------------
def experiment_problem4() -> Dict:
    print("[4/5] 题目四：距离矩阵 / H=2、H=4 / 访问顺序 ...")
    matrix = compute_distance_matrix()
    write_csv("p4_distance_matrix.csv", ["point"] + list(KEY_POINTS),
              [[lab] + matrix[i] for i, lab in enumerate(KEY_POINTS)])
    heatmap(os.path.join(FIGURES, "fig5_p4_distance.svg"),
            "题目四：关键点两两最小移动代价矩阵（A*）",
            list(KEY_POINTS), list(KEY_POINTS), matrix,
            width=700, height=560)

    dist = distances_from(matrix)
    orders = ["natural", "value_desc", "gain_desc", "gain_asc"]
    rows = []
    for depth in (1, 2, 3, 4):
        for alg in ("minimax", "alphabeta"):
            for order in orders:
                target, value, stats, _ = best_first_move(depth, alg, order, dist)
                rows.append([depth, alg, order, target, value, stats.visited,
                             stats.pruned_moves, stats.cutoffs])
    write_csv("p4_search_compare.csv",
              ["depth", "algorithm", "order", "best_target", "best_value",
               "visited", "pruned_moves", "cutoffs"], rows)

    # 根节点候选评估
    ranking = []
    for depth in (2, 4):
        _, _, _, ranked = best_first_move(depth, "alphabeta", "natural", dist)
        for item in ranked:
            ranking.append([depth, item["target"], item["gain"], item["value"]])
    write_csv("p4_root_ranking.csv", ["depth", "target", "gain", "value"], ranking)

    pv2 = principal_variation(2, dist)
    pv4 = principal_variation(4, dist)
    pv_rows = [["H=2"] + [f"{p}:{t}(gain={g},U={u})" for p, t, g, u in pv2],
               ["H=4"] + [f"{p}:{t}(gain={g},U={u})" for p, t, g, u in pv4]]
    write_csv("p4_principal_variation.csv", ["depth", "ply1", "ply2", "ply3", "ply4"],
              pv_rows)

    # 图：不同深度下的访问状态数
    depths = [1, 2, 3, 4]
    def vals(alg, order):
        return [next(r[5] for r in rows if r[0] == d and r[1] == alg and r[2] == order)
                for d in depths]
    grouped_bar_chart(
        os.path.join(FIGURES, "fig6_p4_visited.svg"),
        "题目四：不同搜索深度下的状态访问数（4 个宝藏）",
        [f"H={d}" for d in depths],
        [("Minimax", vals("minimax", "natural"), "#8a8a8a"),
         ("Alpha-Beta（原始顺序）", vals("alphabeta", "natural"), "#3b6ea5"),
         ("Alpha-Beta（收益降序）", vals("alphabeta", "gain_desc"), "#4c9a6a")],
        ylabel="访问状态数", value_fmt="{:.0f}")

    # 图：H=4 时不同顺序的剪枝效果
    h4 = [r for r in rows if r[0] == 4 and r[1] == "alphabeta"]
    grouped_bar_chart(
        os.path.join(FIGURES, "fig7_p4_order.svg"),
        "题目四（H=4）：Alpha-Beta 在不同候选顺序下的访问与剪枝",
        [r[2] for r in h4],
        [("访问状态数", [r[5] for r in h4], "#3b6ea5"),
         ("剪枝着法数", [r[6] for r in h4], "#e08a3c")],
        ylabel="数量", value_fmt="{:.0f}", width=720)

    summary = {
        "H2": next(({"target": r[3], "value": r[4], "visited": r[5]}
                    for r in rows if r[0] == 2 and r[1] == "alphabeta" and r[2] == "natural"), None),
        "H4": next(({"target": r[3], "value": r[4], "visited": r[5]}
                    for r in rows if r[0] == 4 and r[1] == "alphabeta" and r[2] == "natural"), None),
        "pv_H2": pv2, "pv_H4": pv4,
        "root_ranking": ranking,
    }
    return {"matrix": matrix, "rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("实验二对照实验")
    print("=" * 72)
    p1 = experiment_problem1()
    p1b = experiment_weighted_astar()
    p2 = experiment_problem2()
    p3 = experiment_problem3()
    p4 = experiment_problem4()

    summary = {
        "p1_official": p1["official"],
        "p1_density": p1["density_rows"],
        "p1_weighted_astar": {"header": p1b["header"], "rows": p1b["rows"]},
        "p2_official": p2["official"],
        "p2_summary": p2["summary"],
        "p3_official_value": p3["official_value"],
        "p3_variants": p3["variants"],
        "p3_summary": p3["summary"],
        "p4": p4["summary"],
    }
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("P1 官方样例：")
    for o in p1["official"]:
        print(f"  {o['method']:<26} expanded={o['expanded']:<4} path_len={o['path_len']} time={o['time_ms']}ms")
    print("拓展：加权 A*（w, 平均扩展, 平均路径, 路径/最优, 最大比值, 次优数）")
    for r in p1b["rows"]:
        print("  ", r)
    print("P2 官方样例：")
    for o in p2["official"]:
        print(f"  {o['method']:<26} cost={o['cost']} expanded={o['expanded']}")
    print("P2 随机地图汇总：", p2["summary"])
    print("P3 示例树值：", p3["official_value"])
    print("P3 随机树汇总：", p3["summary"])
    print("P4 H=2：", p4["summary"]["H2"])
    print("P4 H=4：", p4["summary"]["H4"])
    print("P4 主变例 H=4：", p4["summary"]["pv_H4"])
    print()
    print("结果已写入:", RESULTS)
    print("图表已写入:", FIGURES)


if __name__ == "__main__":
    main()
