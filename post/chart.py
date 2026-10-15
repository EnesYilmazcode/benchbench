"""The X post image: one bar per benchmark, share of sampled items found broken."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
BG, INK, MUTED, RULE, BAR = "#0e141e", "#e5eaf1", "#95a2b4", "#2a3547", "#f0695c"
SHORT = {"MBPP (sanitized)": "MBPP", "Terminal-Bench 2.0": "Terminal-Bench 2", "HMMT February 2025": "HMMT 2025"}

rows = sorted(json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8")), key=lambda r: -r["rate"])
names = [SHORT.get(r["name"], r["name"]) for r in rows]
vals = [100 * r["rate"] for r in rows]

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

w = 0.68
for i, v in enumerate(vals):
    if v > 0:  # rounded box that starts below the axis, so the clip leaves a square base
        ax.add_patch(FancyBboxPatch((i - w / 2, -2), w, v + 2, boxstyle="round,pad=0,rounding_size=0.06",
                                    mutation_aspect=8, fc=BAR, ec="none", clip_on=True))
    ax.text(i, v + 0.5, f"{v:.0f}%" if v >= 1 or v == 0 else f"{v:.1f}%", ha="center", va="bottom",
            color=INK, fontsize=20, fontweight="bold")

ax.set_xlim(-0.6, len(vals) - 0.4)
ax.set_ylim(0, 25)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor", color=INK, fontsize=19)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_yticklabels([f"{t}%" for t in [0, 5, 10, 15, 20, 25]], color=MUTED, fontsize=17)
ax.tick_params(length=0, pad=8)
ax.grid(axis="y", color=RULE, lw=1)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(0, color=MUTED, lw=1.5)

fig.text(0.055, 0.925, "How much of each AI benchmark is broken", color=INK, fontsize=34, fontweight="bold")
fig.text(0.055, 0.875, "Share of sampled questions and tasks with a wrong answer, an ambiguous question, or a broken grader",
         color=MUTED, fontsize=18)
fig.text(0.975, 0.025, "enesyilmazcode.github.io/benchbench", color=MUTED, fontsize=16, ha="right")
fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.22)
fig.savefig(ROOT / "post/benchbench-chart.png", facecolor=BG)
print("post/benchbench-chart.png")
