# -*- coding: utf-8 -*-
"""轻量级 SVG 图表生成器（不依赖第三方绘图库）。

提供分组柱状图、折线图与热力图，供实验报告使用。
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence

FONT = "Microsoft YaHei, 'PingFang SC', 'SimHei', sans-serif"
PALETTE = ["#3b6ea5", "#e08a3c", "#4c9a6a", "#b05a5a", "#7a6ba8", "#8a8a8a"]


def _esc(text: object) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1
    exp = 10 ** math.floor(math.log10(value))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= m * exp:
            return m * exp
    return 10 * exp


def grouped_bar_chart(path: str, title: str, categories: Sequence[str],
                      series: Sequence[tuple], ylabel: str = "",
                      width: int = 860, height: int = 460,
                      value_fmt: str = "{:.0f}") -> None:
    """绘制分组柱状图。

    series: [(图例名, 数值列表, 颜色), ...]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    left, right, top, bottom = 78, 170, 62, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [v for _, vals, _ in series for v in vals]
    vmax = _nice_max(max(all_vals) if all_vals else 1)
    ticks = 5

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="19" '
        f'font-weight="600" fill="#22303f">{_esc(title)}</text>',
    ]

    # y 轴刻度与网格
    for k in range(ticks + 1):
        val = vmax * k / ticks
        y = top + plot_h - plot_h * k / ticks
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e3e7ec" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="12" fill="#5a6673">{val:g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
                 f'stroke="#8a949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
                 f'y2="{top + plot_h}" stroke="#8a949e" stroke-width="1.2"/>')
    if ylabel:
        parts.append(f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18,{top + plot_h / 2})" '
                     f'text-anchor="middle" font-size="13" fill="#44515e">{_esc(ylabel)}</text>')

    n_cat = max(1, len(categories))
    n_ser = max(1, len(series))
    group_w = plot_w / n_cat
    bar_gap = group_w * 0.16
    bar_w = (group_w - bar_gap * 2) / n_ser

    for ci, cat in enumerate(categories):
        gx = left + ci * group_w
        for si, (label, vals, color) in enumerate(series):
            v = vals[ci] if ci < len(vals) else 0
            h = plot_h * (v / vmax) if vmax else 0
            x = gx + bar_gap + si * bar_w
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.92:.1f}" '
                         f'height="{h:.1f}" fill="{color}" rx="2"/>')
            if v > 0 and bar_w > 22:
                parts.append(f'<text x="{x + bar_w * 0.46:.1f}" y="{y - 5:.1f}" '
                             f'text-anchor="middle" font-size="10" fill="#44515e">'
                             f'{_esc(value_fmt.format(v))}</text>')
        parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{top + plot_h + 22}" '
                     f'text-anchor="middle" font-size="12" fill="#44515e">{_esc(cat)}</text>')

    # 图例
    lx = left + plot_w + 22
    for si, (label, _, color) in enumerate(series):
        ly = top + 8 + si * 26
        parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 21}" y="{ly + 12}" font-size="12.5" fill="#33404d">'
                     f'{_esc(label)}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def line_chart(path: str, title: str, x_labels: Sequence[str],
               series: Sequence[tuple], ylabel: str = "",
               width: int = 860, height: int = 460) -> None:
    """绘制折线图；series: [(图例名, 数值列表, 颜色), ...]。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    left, right, top, bottom = 78, 180, 62, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [v for _, vals, _ in series for v in vals]
    vmax = _nice_max(max(all_vals) if all_vals else 1)
    ticks = 5
    n = max(1, len(x_labels) - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="19" '
        f'font-weight="600" fill="#22303f">{_esc(title)}</text>',
    ]
    for k in range(ticks + 1):
        val = vmax * k / ticks
        y = top + plot_h - plot_h * k / ticks
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e3e7ec"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="12" fill="#5a6673">{val:g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
                 f'stroke="#8a949e"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
                 f'y2="{top + plot_h}" stroke="#8a949e"/>')
    if ylabel:
        parts.append(f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18,{top + plot_h / 2})" '
                     f'text-anchor="middle" font-size="13" fill="#44515e">{_esc(ylabel)}</text>')

    for ci, lab in enumerate(x_labels):
        x = left + plot_w * ci / n if n else left + plot_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" '
                     f'font-size="12" fill="#44515e">{_esc(lab)}</text>')

    for si, (label, vals, color) in enumerate(series):
        pts = []
        for ci, v in enumerate(vals):
            x = left + plot_w * ci / n if n else left + plot_w / 2
            y = top + plot_h - plot_h * (v / vmax if vmax else 0)
            pts.append((x, y))
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#ffffff" '
                         f'stroke="{color}" stroke-width="2"/>')

    lx = left + plot_w + 24
    for si, (label, _, color) in enumerate(series):
        ly = top + 8 + si * 26
        parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 21}" y="{ly + 12}" font-size="12.5" fill="#33404d">'
                     f'{_esc(label)}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def heatmap(path: str, title: str, row_labels: Sequence[str],
            col_labels: Sequence[str], matrix: Sequence[Sequence[float]],
            width: int = 720, height: int = 520,
            value_fmt: str = "{:.0f}") -> None:
    """绘制带数值标注的矩阵热力图（蓝—白—红渐变）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    left, top, right, bottom = 92, 74, 40, 58
    cell_w = (width - left - right) / max(1, len(col_labels))
    cell_h = (height - top - bottom) / max(1, len(row_labels))
    vals = [v for row in matrix for v in row]
    vmax = max(vals) if vals else 1

    def color(v: float) -> str:
        t = v / vmax if vmax else 0
        # 白 -> 浅蓝 -> 深蓝
        r = int(255 - t * 125)
        g = int(255 - t * 105)
        b = 255
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="19" '
        f'font-weight="600" fill="#22303f">{_esc(title)}</text>',
    ]
    for j, lab in enumerate(col_labels):
        parts.append(f'<text x="{left + (j + 0.5) * cell_w:.1f}" y="{top - 14}" '
                     f'text-anchor="middle" font-size="13" fill="#44515e">{_esc(lab)}</text>')
    for i, lab in enumerate(row_labels):
        parts.append(f'<text x="{left - 14}" y="{top + (i + 0.5) * cell_h + 5:.1f}" '
                     f'text-anchor="end" font-size="13" fill="#44515e">{_esc(lab)}</text>')
        for j in range(len(col_labels)):
            v = matrix[i][j]
            x = left + j * cell_w
            y = top + i * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 3:.1f}" '
                         f'height="{cell_h - 3:.1f}" fill="{color(v)}" stroke="#ffffff" rx="3"/>')
            txt_color = "#ffffff" if v / vmax > 0.55 else "#22303f"
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                         f'text-anchor="middle" font-size="14" fill="{txt_color}">'
                         f'{_esc(value_fmt.format(v))}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
