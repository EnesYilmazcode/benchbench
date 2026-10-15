"""The X post image: one bar per benchmark, share of sampled items found broken, with its maker's logo."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LOGOS = ROOT / "post/logos"
BG, INK, MUTED, BASE, BAR = "#ffffff", "#141414", "#6b6b6b", "#c8c8c8", "#d4553f"
SHORT = {"MBPP (sanitized)": "MBPP", "Terminal-Bench 2.0": "Terminal-Bench 2", "HMMT February 2025": "HMMT 2025"}
MAKER = {  # benchmark -> the organisation that published it
    "mbpp": "google.com", "terminal_bench2": "stanford.edu", "swe_verified": "openai.com", "mmlu_pro": "uwaterloo.ca",
    "humaneval": "openai.com", "simpleqa": "openai.com", "tau2": "sierra.ai", "mmlu": "berkeley.edu",
    "gsm8k": "openai.com", "swe_pro": "scale.com", "hellaswag": "allenai.org",
}

rows = sorted(json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8")), key=lambda r: -r["rate"])
rows = [r for r in rows if r["flawed"]]  # 0% bars carry no information on a phone
vals = [100 * r["rate"] for r in rows]

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

w = 0.62
for i, v in enumerate(vals):
    # rounded box that starts below the axis, so the clip leaves a square base
    ax.add_patch(FancyBboxPatch((i - w / 2, -2), w, v + 2, boxstyle="round,pad=0,rounding_size=0.05",
                                mutation_aspect=8, fc=BAR, ec="none"))
    ax.text(i, v + 0.4, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", ha="center", va="bottom",
            color=INK, fontsize=21, fontweight="semibold")

ax.set_xlim(-0.6, len(vals) + 0.05)
ax.set_ylim(0, 23)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(0, color=BASE, lw=1.5)



def logo(domain, box=32):
    """The favicon with its white or transparent margin cut off, fitted into a box x box square."""
    im = Image.open(LOGOS / f"{domain}.png").convert("RGBA")
    flat = Image.new("RGBA", im.size, "white")
    flat.alpha_composite(im)
    ink = flat.convert("L").point(lambda p: 255 if p < 245 else 0).getbbox()
    im = im.crop(ink) if ink else im
    k = box / max(im.size)
    return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)


ANGLE = math.radians(35)
DIR = (math.cos(ANGLE), -math.sin(ANGLE))  # the slant of the labels, in offset points
for i, r in enumerate(rows):
    # logo first, then the name, both on one line that starts under the bar and runs down to the right
    lx, ly = -6, -24
    ax.add_artist(AnnotationBbox(OffsetImage(logo(MAKER[r["bench"]])), (i, 0), xycoords=("data", "axes fraction"),
                                 xybox=(lx, ly), boxcoords="offset points", frameon=False))
    gap = 26
    ax.annotate(SHORT.get(r["name"], r["name"]), (i, 0), xycoords=("data", "axes fraction"),
                xytext=(lx + gap * DIR[0], ly + gap * DIR[1]), textcoords="offset points", rotation=-35,
                rotation_mode="anchor", ha="left", va="center", color=INK, fontsize=18)

fig.text(0.04, 0.9, "What % of each AI benchmark is broken", color=INK, fontsize=36, fontweight="semibold")
fig.text(0.975, 0.03, "enesyilmazcode.github.io/benchbench", color=MUTED, fontsize=15, ha="right")
fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.3)
fig.savefig(ROOT / "post/benchbench-chart.png", facecolor=BG)
print("post/benchbench-chart.png")
