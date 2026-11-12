# 实验二：搜索算法（A\* / Minimax / Alpha-Beta）

当代人工智能实验课 · 实验二。实现并验证了 A\*、加权 A\*、Minimax 与 Alpha-Beta 剪枝，完成迷宫寻路、加权地形、博弈树和宝藏分配四个题目。仓库包含完整代码、单元测试、对照实验脚本与实验结果，核心代码仅依赖 Python 标准库。

## 结果速览

| 题目 | 方法 | 实际输出 | 预期输出 |
| --- | --- | --- | --- |
| 一 · 迷宫寻路 | A\* + 曼哈顿距离 | `[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]` | 同左 |
| 二 · 加权地形 | 加权 A\*（修正曼哈顿） | `7` | `7` |
| 三 · 博弈树 | Alpha-Beta | `5` | `5` |
| 四 · 宝藏分配 | A\* + Alpha-Beta | `H=2 → {"best_target": "T2", "best_value": -1}`<br>`H=4 → {"best_target": "T3", "best_value": 3}` | 同左 |

## 目录结构

```text
.
├── main.py                     # 运行四个题目的官方用例（PASS/FAIL 校验）
├── src/
│   ├── astar.py                # A* 通用框架 + 题目一/二
│   ├── game_tree.py            # 题目三：TreeNode / Minimax / Alpha-Beta
│   ├── treasure_game.py        # 题目四：距离矩阵 + 博弈状态 + 搜索
│   └── run_all.py              # 官方用例定义与输出校验
├── tests/
│   └── test_lab2.py            # 20 个单元测试
├── experiments/
│   ├── run_experiments.py      # 全部对照实验，生成 results/
│   └── chart.py                # 无第三方依赖的 SVG 图表生成器
└── results/                    # 实验数据（CSV / JSON）
```

## 运行方式

要求 Python 3.10+，核心代码只依赖标准库。

```bash
# 1. 运行四个题目的官方用例
python main.py

# 2. 运行单元测试
python -m unittest discover -s tests -v

# 3. 重新生成全部实验数据（固定随机种子，可复现）
python experiments/run_experiments.py
```

## 模块与接口

### 题目一：`src.astar.solve_maze(start, goal, grid) -> List[Coord]`

- 输入：起点、终点 `(row, col)`，二维字符网格，`'X'` 为障碍。
- 输出：从起点到终点的最短路径坐标列表；无解返回 `[]`。
- 默认启发函数为曼哈顿距离；支持 `heuristic=` 与 `tie_break=` 参数用于对照实验。

### 题目二：`src.astar.solve_weighted(start, goal, grid) -> int`

- 输入：起点、终点、地形网格（数字为进入代价，`'S'`/`'E'` 代价为 0，`'X'` 为障碍）。
- 输出：最小总代价；无解返回 `-1`。
- 默认启发函数 `h = max(0, 曼哈顿距离 - 1)`，在终点代价为 0 的规则下严格可采纳。

### 题目三：`src.game_tree.best_score(root, method="alphabeta") -> float`

- 输入：`TreeNode` 根节点（`is_maximizing_player=True` 表示 MAX 层）。
- 输出：MAX 玩家的最佳得分。
- 支持 `build_tree(json_dict)` 从题目给出的 JSON 结构建树；`method="minimax"` 可切换为无剪枝版本。

### 题目四：`src.treasure_game`

- `compute_distance_matrix()`：输出 A、B、T1～T4 的 6×6 最小移动代价矩阵。
- `best_first_move(depth, algorithm="alphabeta", order="natural")`：返回 `(最优首步, 收益差 U, 统计, 各首步评估)`。
- `principal_variation(depth)`：输出主变例。
- `GameState`：不可变状态对象，记录双方位置、剩余宝藏、累计收益与当前玩家。

## 算法要点

- **A\***：`f = g + h`，二叉堆优先队列，允许更优路径重新入队；`larger-g` 平局策略能显著减少开阔网格上的扩展。
- **加权 A\***：进入代价取目标格地形权值，终点代价为 0；对启发函数做了可采纳性修正。
- **Minimax / Alpha-Beta**：MAX/MIN 交替回传，`α ≥ β` 剪枝；支持自定义子节点顺序，用于衡量排序对剪枝效率的影响。
- **综合题**：先用 A\* 预计算关键点距离矩阵，再以 `(posA, posB, remaining, scoreA, scoreB, turn)` 为状态做深度受限的 Alpha-Beta 搜索。

## 实验与结论

`experiments/run_experiments.py` 会生成 `results/` 下的 CSV/JSON，核心结论：

| 实验 | 结论 |
| --- | --- |
| 迷宫：h=0 vs 曼哈顿 | 障碍密度 0.10→0.30 时，曼哈顿减少扩展 11.1%→40.6% |
| 迷宫：larger-g 平局 | 扩展数降至 FIFO 的 15%～32%，路径仍最优 |
| 迷宫：加权 A\* | w=1.5 时扩展降 73.9%、路径仅长 3.5%；w≥2 收益饱和 |
| 加权地形：三种启发 | 地形权值远大于 1 时曼哈顿偏弱，仅省约 4% 扩展；代价 0 误差 |
| Alpha-Beta：随机树 | 原始顺序少访问 30.1%，完美排序少访问 59.7%，最差顺序仅 1.6% |
| 宝藏分配：H=2 vs H=4 | H=2 选 T2（U=−1），H=4 选 T3（U=3），体现地平线效应 |
| 宝藏分配：排序 | H=4 时收益降序把访问状态从 64 降到 48（−25%） |

## 可复现性

- 全部随机实验使用固定种子 `20261112`；
- 迷宫实验先检查连通性，并断言不同策略的路径长度一致；
- 博弈树实验用 200 棵随机树验证 Alpha-Beta 与 Minimax 结果相同；
- `python main.py` 与 `python -m unittest discover -s tests -v` 均可在无第三方依赖的环境下运行。
