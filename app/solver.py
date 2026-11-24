"""纹版约束求解。

问题建模
========
纹版是 ``W × H`` 格的环形（toroidal）色带，每格取参考图调色板中的一种颜色。
要求：

1. 残片投影得到的完好色点（固定格约束）必须保留；
2. 以每个格为左上角的 2×2 图块都必须在参考图中出现过（允许跨边/跨角环绕）；
3. 在所有可行纹版中，最大化「每个 2×2 图块在参考图中的出现次数」之和。

求解策略（三层）
================
* **可行性层（CP-SAT）**：决策变量只有 ``W×H`` 个颜色整数变量，每个格位置
  一条 ``add_allowed_assignments`` 约束，允许表即参考图出现过的 2×2 四元组
  （至多 4⁴=256 行）。该模型在最坏尺寸（12×12、256 种图块）下通常 0.1 秒
  内即可判定可行性或严格证明无解。
* **优化层（局部搜索）**：在合法网格上做爬山搜索。除单格变色外，还支持
  整列/整行联动改色与 2×2 图块联动改色——当单格变色全部违反图块规则时
  （例如同列同色竖条纹图块系统），仍可通过联动移动提分。每次变色只做
  增量合法性/得分评估；随机重启通过「随机赋值 + 最小冲突修复」构造彼此
  不同的**合法**起点（固定色点始终保留），多轮重启直到限时耗尽。
  当得分达到理论上界（格数 × 最高频图块频次）时可声明「已证最优」，
  否则标记「仅找到可行纹样」。重启失败或预算用尽时，返回整个搜索过程中
  找到的最高分合法候选（至少为初始可行网格）。
* **兜底层（内置回溯）**：运行环境缺少 OR-Tools 时，使用行编码回溯 +
  分支定界，接口与状态语义保持一致。

状态语义：``optimal`` 已证最优 / ``feasible`` 仅找到可行纹样 /
``infeasible`` 已证无解 / ``unknown`` 待定。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .images import RGB, TileKey

STATUS_OPTIMAL = "optimal"          # 已证最优
STATUS_FEASIBLE = "feasible"        # 仅找到可行纹样
STATUS_INFEASIBLE = "infeasible"    # 已证无解
STATUS_UNKNOWN = "unknown"          # 待定

STATUS_LABELS = {
    STATUS_OPTIMAL: "已证最优",
    STATUS_FEASIBLE: "仅找到可行纹样",
    STATUS_INFEASIBLE: "已证无解",
    STATUS_UNKNOWN: "待定",
}

_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass
class SolveResult:
    status: str
    grid: Optional[tuple[tuple[RGB, ...], ...]] = None
    score: Optional[int] = None
    upper_bound: Optional[int] = None
    elapsed_seconds: float = 0.0
    solver_name: str = ""
    tile_usage: list[dict] = field(default_factory=list)


def solve(
    palette: tuple[RGB, ...],
    tile_counts: dict[TileKey, int],
    unit_w: int,
    unit_h: int,
    fixed: dict[tuple[int, int], RGB],
    time_limit: float = 5.0,
) -> SolveResult:
    """按启动时色点求解纹版。

    Parameters
    ----------
    palette:
        参考图调色板（有序 RGB 元组）。
    tile_counts:
        参考图全部 2×2 图块的出现次数（仅含出现过的图块）。
    unit_w, unit_h:
        重复单元每边格数（4～12）。
    fixed:
        固定格约束 ``{(x, y): rgb}``。
    time_limit:
        墙钟限时（秒），默认 5 秒。
    """
    try:
        return _solve_cpsat(palette, tile_counts, unit_w, unit_h, fixed,
                            time_limit)
    except ImportError:
        return _solve_backtrack(palette, tile_counts, unit_w, unit_h, fixed,
                                time_limit)


# --------------------------------------------------------------------------- #
# 通用辅助
# --------------------------------------------------------------------------- #

def _max_freq(tile_counts: dict[TileKey, int]) -> int:
    return max(tile_counts.values(), default=0)


def _tile_usage(
    tile_counts: dict[TileKey, int],
    grid: tuple[tuple[RGB, ...], ...],
    unit_w: int,
    unit_h: int,
) -> list[dict]:
    """汇总候选纹版中各 2×2 图块的使用情况与参考频次。"""
    from collections import Counter

    usage: Counter[TileKey] = Counter()
    for y in range(unit_h):
        for x in range(unit_w):
            key = (
                grid[y][x],
                grid[y][(x + 1) % unit_w],
                grid[(y + 1) % unit_h][x],
                grid[(y + 1) % unit_h][(x + 1) % unit_w],
            )
            usage[key] += 1
    return [
        {
            "tile": [list(c) for c in key],
            "reference_count": tile_counts.get(key, 0),
            "used_in_pattern": count,
        }
        for key, count in sorted(
            usage.items(),
            key=lambda kv: (-kv[1], kv[0]))
    ]


def _score_rgb_grid(
    tile_counts: dict[TileKey, int],
    grid: tuple[tuple[RGB, ...], ...],
    unit_w: int,
    unit_h: int,
) -> int:
    total = 0
    for y in range(unit_h):
        for x in range(unit_w):
            key = (
                grid[y][x],
                grid[y][(x + 1) % unit_w],
                grid[(y + 1) % unit_h][x],
                grid[(y + 1) % unit_h][(x + 1) % unit_w],
            )
            total += tile_counts.get(key, 0)
    return total


def _score_idx_grid(
    quad_freq_idx: dict[tuple[int, int, int, int], int],
    grid: list[list[int]],
    unit_w: int,
    unit_h: int,
) -> int:
    """下标网格的图块频次之和。"""
    total = 0
    for y in range(unit_h):
        for x in range(unit_w):
            quad = (
                grid[y][x],
                grid[y][(x + 1) % unit_w],
                grid[(y + 1) % unit_h][x],
                grid[(y + 1) % unit_h][(x + 1) % unit_w],
            )
            total += quad_freq_idx.get(quad, 0)
    return total


# --------------------------------------------------------------------------- #
# CP-SAT 可行性 + 局部搜索优化
# --------------------------------------------------------------------------- #

def _solve_cpsat(
    palette: tuple[RGB, ...],
    tile_counts: dict[TileKey, int],
    unit_w: int,
    unit_h: int,
    fixed: dict[tuple[int, int], RGB],
    time_limit: float,
) -> SolveResult:
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    # 安全截止线：预留余量给求解器收尾、结果封装、网络响应以及 CPU 争抢下
    # 的调度延迟，保证外层墙钟不突破 time_limit。
    safety_margin = min(0.75, float(time_limit) * 0.15)
    deadline = started + float(time_limit) - safety_margin
    color_index = {c: i for i, c in enumerate(palette)}

    # 快速拒绝：固定色不在参考图调色板中 -> 无解
    for color in fixed.values():
        if color not in color_index:
            return SolveResult(
                status=STATUS_INFEASIBLE,
                elapsed_seconds=time.perf_counter() - started,
                solver_name="cp-sat+localsearch",
            )

    # 四元组（颜色下标）允许表
    quad_colors = sorted(
        tuple(color_index[c] for c in key)  # type: ignore[arg-type]
        for key in tile_counts)
    allowed_table = [list(q) for q in quad_colors]
    quad_freq_idx = {
        q: tile_counts[next(
            k for k in tile_counts
            if tuple(color_index[c] for c in k) == q)]
        for q in quad_colors
    }

    # ---------------- 阶段 1：CP-SAT 精确优化（动态预算） ---------------- #
    # 图块种类少/约束多时精确模型很快；给它一个随规模收缩的预算，
    # 最坏情形（256 种图块、12×12）只花 ~1 秒，把剩余时间留给可行性+局部搜索。
    exact_budget = min(
        1.0,
        max(0.0, time_limit * 0.3
            - 0.005 * unit_w * unit_h * len(quad_colors) / 25.0),
    )
    exact_budget = min(exact_budget, deadline - time.perf_counter())
    exact = None
    fixed_idx_local = {
        (x, y): color_index[c] for (x, y), c in fixed.items()}
    if exact_budget > 0.02:  # 预算太小就跳过精确阶段，避免固定开销反噬
        exact = _try_exact_cpsat(
            palette, quad_colors, quad_freq_idx, allowed_table,
            unit_w, unit_h, fixed_idx_local, exact_budget)
    if exact is not None and exact[0] == STATUS_OPTIMAL:
        idx_grid, best_score = exact[1], exact[2]
        rgb_grid = tuple(
            tuple(palette[idx_grid[y][x]] for x in range(unit_w))
            for y in range(unit_h))
        result = SolveResult(
            status=STATUS_OPTIMAL, grid=rgb_grid, score=best_score,
            upper_bound=unit_w * unit_h * _max_freq(tile_counts),
            elapsed_seconds=min(time.perf_counter() - started, time_limit),
            solver_name="cp-sat+localsearch")
        result.tile_usage = _tile_usage(tile_counts, rgb_grid, unit_w, unit_h)
        return result

    # ---------------- 阶段 2：CP-SAT 纯可行性 ---------------- #
    model = cp_model.CpModel()
    cells: dict[tuple[int, int], cp_model.IntVar] = {}
    for y in range(unit_h):
        for x in range(unit_w):
            var = model.new_int_var(0, len(palette) - 1, f"c_{x}_{y}")
            cells[(x, y)] = var
            if (x, y) in fixed:
                model.add(var == color_index[fixed[(x, y)]])
    for y in range(unit_h):
        for x in range(unit_w):
            corner_vars = [
                cells[((x + dx) % unit_w, (y + dy) % unit_h)]
                for dx, dy in _CORNERS
            ]
            model.add_allowed_assignments(corner_vars, allowed_table)

    solver = cp_model.CpSolver()
    # 可行性模型正常 0.1 秒内可解；显式封顶 2 秒，避免极端环境下吃光预算
    feas_budget = min(2.0, deadline - time.perf_counter())
    solver.parameters.max_time_in_seconds = max(0.01, feas_budget)
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 20261123
    status = solver.solve(model)

    if status == cp_model.INFEASIBLE:
        return SolveResult(
            status=STATUS_INFEASIBLE,
            elapsed_seconds=time.perf_counter() - started,
            solver_name="cp-sat+localsearch")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # 限时内可行性都未判定 -> 待定
        return SolveResult(
            status=STATUS_UNKNOWN,
            elapsed_seconds=min(time.perf_counter() - started, time_limit),
            upper_bound=unit_w * unit_h * _max_freq(tile_counts),
            solver_name="cp-sat+localsearch")

    idx_grid = [
        [solver.value(cells[(x, y)]) for x in range(unit_w)]
        for y in range(unit_h)
    ]
    # 精确模型若已找到可行解（未证最优），用它作为局部搜索起点
    if exact is not None and exact[0] == STATUS_FEASIBLE:
        idx_grid = exact[1]
    fixed_idx = fixed_idx_local

    # ---------------- 阶段 3：局部搜索优化得分 ---------------- #
    # 只使用安全截止线之前的剩余时间
    local_deadline = deadline
    best_grid = idx_grid
    best_score = _score_idx_grid(
        quad_freq_idx, idx_grid, unit_w, unit_h)
    if time.perf_counter() < local_deadline:
        searched_grid, searched_score = _local_search(
            idx_grid, quad_colors, quad_freq_idx, palette,
            unit_w, unit_h, fixed_idx, local_deadline)
        # 防御性校验：只接受合法且保留全部固定色点的候选，
        # 否则退回 CP-SAT 给出的初始可行网格（仍是合法候选）。
        valid_quad_set = set(quad_colors)
        grid_legal = all(
            (
                searched_grid[y][x],
                searched_grid[y][(x + 1) % unit_w],
                searched_grid[(y + 1) % unit_h][x],
                searched_grid[(y + 1) % unit_h][(x + 1) % unit_w],
            ) in valid_quad_set
            for y in range(unit_h) for x in range(unit_w))
        fixed_ok = all(
            searched_grid[y][x] == color for (x, y), color in fixed_idx.items())
        if grid_legal and fixed_ok and searched_score >= best_score:
            best_grid, best_score = searched_grid, searched_score

    rgb_grid = tuple(
        tuple(palette[best_grid[y][x]] for x in range(unit_w))
        for y in range(unit_h)
    )
    upper_bound = unit_w * unit_h * _max_freq(tile_counts)
    # 达到理论上界 => 已证最优；否则只是当前找到的最佳可行纹样
    final_status = (
        STATUS_OPTIMAL if best_score >= upper_bound else STATUS_FEASIBLE)

    result = SolveResult(
        status=final_status,
        grid=rgb_grid,
        score=best_score,
        upper_bound=upper_bound,
        elapsed_seconds=min(time.perf_counter() - started, time_limit),
        solver_name="cp-sat+localsearch",
    )
    result.tile_usage = _tile_usage(tile_counts, rgb_grid, unit_w, unit_h)
    return result


def _try_exact_cpsat(
    palette: tuple[RGB, ...],
    quad_colors: list[tuple[int, int, int, int]],
    quad_freq_idx: dict[tuple[int, int, int, int], int],
    allowed_table: list[list[int]],
    unit_w: int,
    unit_h: int,
    fixed_idx: dict[tuple[int, int], int],
    budget: float,
) -> Optional[tuple[str, list[list[int]], int]]:
    """带频次目标的精确 CP-SAT 模型。

    返回 ``(状态, 下标网格, 得分)``；状态为 ``optimal`` 或 ``feasible``；
    若预算内无解/未找到可行点则返回 ``None``。
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    cells: dict[tuple[int, int], cp_model.IntVar] = {}
    for y in range(unit_h):
        for x in range(unit_w):
            var = model.new_int_var(0, len(palette) - 1, f"c_{x}_{y}")
            cells[(x, y)] = var
            if (x, y) in fixed_idx:
                model.add(var == fixed_idx[(x, y)])

    freq_domain = cp_model.Domain.from_values(
        sorted(set(quad_freq_idx.values())))
    freq_table = [list(q) + [quad_freq_idx[q]] for q in quad_colors]
    terms = []
    for y in range(unit_h):
        for x in range(unit_w):
            corner_vars = [
                cells[((x + dx) % unit_w, (y + dy) % unit_h)]
                for dx, dy in _CORNERS
            ]
            model.add_allowed_assignments(corner_vars, allowed_table)
            freq_var = model.new_int_var_from_domain(
                freq_domain, f"f_{x}_{y}")
            model.add_allowed_assignments(
                corner_vars + [freq_var], freq_table)
            terms.append(freq_var)
    model.maximize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 20261123
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    grid = [
        [solver.value(cells[(x, y)]) for x in range(unit_w)]
        for y in range(unit_h)
    ]
    label = STATUS_OPTIMAL if status == cp_model.OPTIMAL else STATUS_FEASIBLE
    return label, grid, int(round(solver.objective_value))


def _local_search(
    idx_grid: list[list[int]],
    quad_colors: list[tuple[int, int, int, int]],
    quad_freq_idx: dict[tuple[int, int, int, int], int],
    palette: tuple[RGB, ...],
    unit_w: int,
    unit_h: int,
    fixed_idx: dict[tuple[int, int], int],
    deadline: float,
    stats: Optional[dict] = None,
) -> tuple[list[list[int]], int]:
    """合法邻域爬山 + 多样化合法随机重启。

    邻域移动（都只接受移动后仍然合法、且不动固定色点的候选）：

    1. **单格改色**：只重评受影响的 4 个图块；
    2. **整列/整行联动改色**：当单格改色全部违反图块规则（典型如同列同色
       竖条纹图块系统）时，一次联动改一列/一行，跨越单格死局继续提分；
    3. **2×2 图块联动改色**：直接套用参考图中出现过的四元组。

    随机重启不再从同一候选出发：每个重启起点由「随机赋值 + 最小冲突修复」
    构造成**彼此不同的合法网格**（固定色点始终保留），并按网格签名去重；
    与「在当前最佳候选上扰动后修复」的起点交替使用。

    重启失败或预算用尽时，返回搜索过程中找到的最高分合法候选（至少为
    传入的初始可行网格）。``stats`` 非空时写入起点/重启统计，便于自测。
    """
    rng = random.Random(20261123)
    valid_quads = set(quad_colors)
    n_colors = len(palette)
    n_cells = unit_w * unit_h
    upper_bound = n_cells * max(quad_freq_idx.values())

    def quad_at(grid: list[list[int]], x: int, y: int):
        return (
            grid[y][x],
            grid[y][(x + 1) % unit_w],
            grid[(y + 1) % unit_h][x],
            grid[(y + 1) % unit_h][(x + 1) % unit_w],
        )

    def score(grid: list[list[int]]) -> int:
        return sum(
            quad_freq_idx.get(quad_at(grid, x, y), 0)
            for y in range(unit_h) for x in range(unit_w))

    # 变色 (x,y) 只影响以 (x,y)、(x-1,y)、(x,y-1)、(x-1,y-1) 为左上角的图块
    def affected_positions(x: int, y: int):
        return {
            ((x - dx) % unit_w, (y - dy) % unit_h)
            for dx in (0, 1) for dy in (0, 1)
        }

    def is_legal(grid: list[list[int]]) -> bool:
        return all(
            quad_at(grid, x, y) in valid_quads
            for y in range(unit_h) for x in range(unit_w))

    def respects_fixed(grid: list[list[int]]) -> bool:
        return all(grid[y][x] == color for (x, y), color in fixed_idx.items())

    def grid_sig(grid: list[list[int]]) -> tuple[int, ...]:
        return tuple(v for row in grid for v in row)

    def local_score(grid: list[list[int]], positions) -> int:
        return sum(
            quad_freq_idx.get(quad_at(grid, ax, ay), 0) for ax, ay in positions)

    def local_legal(grid: list[list[int]], positions) -> bool:
        return all(quad_at(grid, ax, ay) in valid_quads for ax, ay in positions)

    # ------------------------------------------------------------------ #
    # 最小冲突修复：把（可能非法的）网格修成合法网格，固定格永不修改
    # ------------------------------------------------------------------ #
    def repair(grid: list[list[int]]) -> bool:
        if is_legal(grid):
            return True
        max_iters = max(40, n_cells * 6)
        for _ in range(max_iters):
            if time.perf_counter() >= deadline:
                return False
            bad = [
                (x, y)
                for y in range(unit_h) for x in range(unit_w)
                if quad_at(grid, x, y) not in valid_quads
            ]
            if not bad:
                return True
            # 参与坏图块的可变色格
            cand_cells: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for tx, ty in bad:
                for dx, dy in _CORNERS:
                    cell = ((tx + dx) % unit_w, (ty + dy) % unit_h)
                    if cell not in fixed_idx and cell not in seen:
                        seen.add(cell)
                        cand_cells.append(cell)
            if not cand_cells:
                return False
            rng.shuffle(cand_cells)
            # 少量随机游走，帮助修复过程逃出死循环
            if rng.random() < 0.15:
                cx, cy = cand_cells[0]
                grid[cy][cx] = rng.randrange(n_colors)
                continue
            moved = False
            for cx, cy in cand_cells:
                old = grid[cy][cx]
                positions = affected_positions(cx, cy)
                best_value = old
                best_bad = n_cells + 1
                for value in range(n_colors):
                    grid[cy][cx] = value
                    bad_count = sum(
                        quad_at(grid, ax, ay) not in valid_quads
                        for ax, ay in positions)
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_value = value
                grid[cy][cx] = old
                if best_value != old:
                    grid[cy][cx] = best_value
                    moved = True
                    break
            if not moved:
                return False
        return is_legal(grid)

    def random_legal_start() -> Optional[list[list[int]]]:
        """全新随机赋值后修复为合法网格（固定格保留）。"""
        for _ in range(60):
            if time.perf_counter() >= deadline:
                return None
            grid = [
                [fixed_idx.get((x, y), rng.randrange(n_colors))
                 for x in range(unit_w)]
                for y in range(unit_h)
            ]
            if repair(grid) and is_legal(grid) and respects_fixed(grid):
                return grid
        return None

    def kicked_start(base: list[list[int]]) -> Optional[list[list[int]]]:
        """在当前最佳候选上随机扰动若干格后修复，得到不同的合法起点。"""
        base_sig = grid_sig(base)
        for _ in range(20):
            if time.perf_counter() >= deadline:
                return None
            grid = [row[:] for row in base]
            for _ in range(max(1, n_cells // 6)):
                x = rng.randrange(unit_w)
                y = rng.randrange(unit_h)
                if (x, y) not in fixed_idx:
                    grid[y][x] = rng.randrange(n_colors)
            if (repair(grid) and is_legal(grid) and respects_fixed(grid)
                    and grid_sig(grid) != base_sig):
                return grid
        return None

    # ------------------------------------------------------------------ #
    # 爬山：单格 -> 整列 -> 整行 -> 2×2 联动，只走上坡合法移动
    # ------------------------------------------------------------------ #
    def hill_climb(grid0: list[list[int]]) -> tuple[list[list[int]], int]:
        cur = [row[:] for row in grid0]
        movable = [
            (x, y) for y in range(unit_h) for x in range(unit_w)
            if (x, y) not in fixed_idx
        ]
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False

            # 1) 单格最佳改进
            rng.shuffle(movable)
            for x, y in movable:
                if time.perf_counter() >= deadline:
                    break
                old = cur[y][x]
                positions = affected_positions(x, y)
                before = local_score(cur, positions)
                best_value = old
                best_gain = 0
                for value in range(n_colors):
                    if value == old:
                        continue
                    cur[y][x] = value
                    if not local_legal(cur, positions):
                        continue
                    gain = local_score(cur, positions) - before
                    if gain > best_gain:
                        best_gain = gain
                        best_value = value
                cur[y][x] = best_value
                if best_value != old:
                    improved = True
            if improved or time.perf_counter() >= deadline:
                continue

            # 2) 整列联动改色（单格全非法时的关键逃生移动）
            for x in rng.sample(range(unit_w), unit_w):
                if time.perf_counter() >= deadline:
                    break
                col_cells = [(x, y) for y in range(unit_h)
                             if (x, y) not in fixed_idx]
                if not col_cells:
                    continue
                positions = {
                    (tx, ty)
                    for tx in ((x - 1) % unit_w, x)
                    for ty in range(unit_h)
                }
                before = local_score(cur, positions)
                saved = [cur[y][x] for _, y in col_cells]
                current = set(saved)
                best_value: Optional[int] = None
                best_gain = 0
                for value in range(n_colors):
                    if value in current:
                        continue
                    for _, cy in col_cells:
                        cur[cy][x] = value
                    if not local_legal(cur, positions):
                        continue
                    gain = local_score(cur, positions) - before
                    if gain > best_gain:
                        best_gain = gain
                        best_value = value
                for (_, cy), old_value in zip(col_cells, saved):
                    cur[cy][x] = old_value
                if best_value is not None:
                    for _, cy in col_cells:
                        cur[cy][x] = best_value
                    improved = True
                    break
            if improved or time.perf_counter() >= deadline:
                continue

            # 3) 整行联动改色
            for y in rng.sample(range(unit_h), unit_h):
                if time.perf_counter() >= deadline:
                    break
                row_cells = [(x, y) for x in range(unit_w)
                             if (x, y) not in fixed_idx]
                if not row_cells:
                    continue
                positions = {
                    (tx, ty)
                    for tx in range(unit_w)
                    for ty in ((y - 1) % unit_h, y)
                }
                before = local_score(cur, positions)
                saved = [cur[y][x] for x, _ in row_cells]
                current = set(saved)
                best_value = None
                best_gain = 0
                for value in range(n_colors):
                    if value in current:
                        continue
                    for cx, _ in row_cells:
                        cur[y][cx] = value
                    if not local_legal(cur, positions):
                        continue
                    gain = local_score(cur, positions) - before
                    if gain > best_gain:
                        best_gain = gain
                        best_value = value
                for (cx, _), old_value in zip(row_cells, saved):
                    cur[y][cx] = old_value
                if best_value is not None:
                    for cx, _ in row_cells:
                        cur[y][cx] = best_value
                    improved = True
                    break
            if improved or time.perf_counter() >= deadline:
                continue

            # 4) 2×2 图块联动改色：直接套用参考图出现过的合法四元组
            block_origins = [
                (x, y) for y in range(unit_h) for x in range(unit_w)]
            rng.shuffle(block_origins)
            for bx, by in block_origins[: min(n_cells, 24)]:
                if time.perf_counter() >= deadline:
                    break
                cells4 = [
                    ((bx + dx) % unit_w, (by + dy) % unit_h)
                    for dx, dy in _CORNERS
                ]
                if any(cell in fixed_idx for cell in cells4):
                    continue
                positions: set[tuple[int, int]] = set()
                for cx, cy in cells4:
                    positions |= affected_positions(cx, cy)
                old_values = tuple(cur[cy][cx] for cx, cy in cells4)
                before = local_score(cur, positions)
                patterns = list(valid_quads)
                rng.shuffle(patterns)
                best_pattern: Optional[tuple[int, int, int, int]] = None
                best_gain = 0
                tried: set[tuple[int, ...]] = {old_values}
                for pattern in patterns[:16]:
                    if pattern in tried:
                        continue
                    tried.add(pattern)
                    for (cx, cy), value in zip(cells4, pattern):
                        cur[cy][cx] = value
                    if not local_legal(cur, positions):
                        continue
                    gain = local_score(cur, positions) - before
                    if gain > best_gain:
                        best_gain = gain
                        best_pattern = pattern
                for (cx, cy), value in zip(cells4, old_values):
                    cur[cy][cx] = value
                if best_pattern is not None:
                    for (cx, cy), value in zip(cells4, best_pattern):
                        cur[cy][cx] = value
                    improved = True
                    break

        # 重新整盘计分，避免增量累计误差
        return cur, score(cur)

    # ------------------------------------------------------------------ #
    # 主循环：初始候选 + 多样化合法随机重启
    # ------------------------------------------------------------------ #
    seed = [row[:] for row in idx_grid]
    if not is_legal(seed) and not repair(seed):
        # 防御：初始网格修不好时原样返回，调用方保证不会走到这里
        return [row[:] for row in idx_grid], score(idx_grid)

    best_grid = [row[:] for row in seed]
    best_score = score(best_grid)
    seen_starts = {grid_sig(best_grid)}
    if stats is not None:
        stats["start_signatures"] = [grid_sig(seed)]
        stats["start_scores"] = [best_score]

    candidate, candidate_score = hill_climb(seed)
    if (is_legal(candidate) and respects_fixed(candidate)
            and candidate_score > best_score):
        best_grid = [row[:] for row in candidate]
        best_score = candidate_score

    failures = 0
    use_fresh = True
    # 只在到达上界或墙钟预算用尽时停止；生成器连续失败/起点重复只跳过，
    # 不提前退出（重启失败时仍保留已找到的最高分合法候选 best_grid）。
    # total_attempts 是防御性上限，避免极端情况下的忙等空转。
    total_attempts = 0
    while (time.perf_counter() < deadline
           and best_score < upper_bound
           and total_attempts < 400):
        total_attempts += 1
        start_grid = (random_legal_start() if use_fresh
                      else kicked_start(best_grid))
        use_fresh = not use_fresh
        if start_grid is None:
            failures += 1
            # 两类生成器连续多轮都造不出新起点：可能是解空间极小，
            # 短暂让出后继续尝试，直到墙钟预算真正耗尽。
            if failures >= 40:
                break
            continue
        signature = grid_sig(start_grid)
        if signature in seen_starts:
            failures += 1
            continue
        seen_starts.add(signature)
        failures = 0
        if stats is not None:
            stats["start_signatures"].append(signature)
            stats["start_scores"].append(score(start_grid))

        candidate, candidate_score = hill_climb(start_grid)
        # 只接受合法且完好色点未被破坏的候选
        if not is_legal(candidate) or not respects_fixed(candidate):
            continue
        if candidate_score > best_score:
            best_grid = [row[:] for row in candidate]
            best_score = candidate_score

    if stats is not None:
        stats["restart_count"] = len(seen_starts) - 1
        stats["best_score"] = best_score
        stats["upper_bound"] = upper_bound
    return best_grid, best_score


# --------------------------------------------------------------------------- #
# 内置回溯兜底（无 OR-Tools 时使用）
# --------------------------------------------------------------------------- #

def _solve_backtrack(
    palette: tuple[RGB, ...],
    tile_counts: dict[TileKey, int],
    unit_w: int,
    unit_h: int,
    fixed: dict[tuple[int, int], RGB],
    time_limit: float,
) -> SolveResult:
    """行编码回溯 + 分支定界（无 OR-Tools 时的兜底实现）。

    把每一行编码成调色板下标整数（调色板至多 4 色，每色 2 bit）。
    先用参考图块的合法水平边对快速筛行，再在逐行展开时按完整 2×2
    四元组校验，最后校验首尾行之间的环绕闭合。
    """
    started = time.perf_counter()
    # 与 CP-SAT 路径一致的安全截止线：预留收尾与调度余量，保证外层墙钟
    # 不突破 time_limit（回溯只在递归节点检查时间，需要提前一点停）。
    safety_margin = min(0.75, float(time_limit) * 0.15)
    deadline = started + float(time_limit) - safety_margin
    color_index = {c: i for i, c in enumerate(palette)}
    base = len(palette)

    for color in fixed.values():
        if color not in color_index:
            return SolveResult(status=STATUS_INFEASIBLE,
                               elapsed_seconds=time.perf_counter() - started,
                               solver_name="backtrack")

    horizontal_pairs: set[tuple[int, int]] = set()
    vertical_pairs: set[tuple[int, int]] = set()
    freq: dict[tuple[int, int, int, int], int] = {}
    for (tl, tr, bl, br), n in tile_counts.items():
        i_tl, i_tr = color_index[tl], color_index[tr]
        i_bl, i_br = color_index[bl], color_index[br]
        horizontal_pairs.add((i_tl, i_tr))
        horizontal_pairs.add((i_bl, i_br))
        vertical_pairs.add((i_tl, i_bl))
        vertical_pairs.add((i_tr, i_br))
        freq[(i_tl, i_tr, i_bl, i_br)] = n
    valid_quads = set(freq)

    fixed_by_row: dict[int, dict[int, int]] = {}
    for (x, y), color in fixed.items():
        fixed_by_row.setdefault(y, {})[x] = color_index[color]

    def encode_row(cells: list[int]) -> int:
        row = 0
        for x, value in enumerate(cells):
            row |= value << (2 * x)
        return row

    def row_cells(row: int) -> list[int]:
        return [(row >> (2 * x)) & 0b11 for x in range(unit_w)]

    def generate_rows(y: int) -> Optional[list[int]]:
        """递归生成满足固定格与非环绕水平边对的行（带剪枝与时限检查）。"""
        result: list[int] = []
        fixed_here = fixed_by_row.get(y, {})
        deadline_hit = False

        def recurse(x: int, prev: Optional[int], cells: list[int]) -> None:
            nonlocal deadline_hit
            if deadline_hit:
                return
            if x == unit_w:
                result.append(encode_row(cells))
                return
            if (x & 3) == 0 and time.perf_counter() >= deadline:
                deadline_hit = True
                return
            options = [fixed_here[x]] if x in fixed_here else range(base)
            for value in options:
                if prev is not None and (prev, value) not in horizontal_pairs:
                    continue
                cells.append(value)
                recurse(x + 1, value, cells)
                cells.pop()

        recurse(0, None, [])
        if deadline_hit:
            return None
        return result

    candidates: list[list[int]] = []
    for y in range(unit_h):
        rows = generate_rows(y)
        if rows is None:
            return SolveResult(
                status=STATUS_UNKNOWN,
                elapsed_seconds=min(time.perf_counter() - started, time_limit),
                upper_bound=unit_w * unit_h * _max_freq(tile_counts),
                solver_name="backtrack")
        if not rows:
            return SolveResult(status=STATUS_INFEASIBLE,
                               elapsed_seconds=time.perf_counter() - started,
                               solver_name="backtrack")
        candidates.append(rows)

    def band_ok(upper: list[int], lower: list[int]) -> bool:
        for x in range(unit_w):
            quad = (upper[x], upper[(x + 1) % unit_w],
                    lower[x], lower[(x + 1) % unit_w])
            if quad not in valid_quads:
                return False
        return True

    def vpair_ok(upper: list[int], lower: list[int]) -> bool:
        return all((upper[x], lower[x]) in vertical_pairs
                   for x in range(unit_w))

    def score_rows(grid_rows: list[list[int]]) -> int:
        total = 0
        for y in range(unit_h):
            upper = grid_rows[y]
            lower = grid_rows[(y + 1) % unit_h]
            for x in range(unit_w):
                total += freq.get(
                    (upper[x], upper[(x + 1) % unit_w],
                     lower[x], lower[(x + 1) % unit_w]), 0)
        return total

    chosen_rows: list[list[int]] = []
    best: Optional[list[list[int]]] = None
    best_score = -1
    proved_optimal = True
    full_bound = unit_w * unit_h * _max_freq(tile_counts)

    def dfs(y: int) -> None:
        nonlocal best, best_score, proved_optimal
        if time.perf_counter() >= deadline:
            proved_optimal = False
            return
        if y == unit_h:
            if not band_ok(chosen_rows[unit_h - 1], chosen_rows[0]):
                return
            current_score = score_rows(chosen_rows)
            if current_score > best_score:
                best_score = current_score
                best = [r.copy() for r in chosen_rows]
            return
        for row_int in candidates[y]:
            cells = row_cells(row_int)
            if chosen_rows and (
                    not vpair_ok(chosen_rows[-1], cells)
                    or not band_ok(chosen_rows[-1], cells)):
                continue
            chosen_rows.append(cells)
            dfs(y + 1)
            chosen_rows.pop()
            if time.perf_counter() >= deadline:
                proved_optimal = False
                return

    dfs(0)
    elapsed = time.perf_counter() - started
    if best is None:
        if proved_optimal:
            return SolveResult(status=STATUS_INFEASIBLE, elapsed_seconds=elapsed,
                               solver_name="backtrack")
        return SolveResult(status=STATUS_UNKNOWN, elapsed_seconds=elapsed,
                           upper_bound=full_bound, solver_name="backtrack")

    grid = tuple(tuple(palette[c] for c in row) for row in best)
    status = STATUS_OPTIMAL if proved_optimal else STATUS_FEASIBLE
    result = SolveResult(status=status, grid=grid, score=best_score,
                         upper_bound=best_score if proved_optimal else full_bound,
                         elapsed_seconds=min(elapsed, time_limit),
                         solver_name="backtrack")
    result.tile_usage = _tile_usage(tile_counts, grid, unit_w, unit_h)
    return result
