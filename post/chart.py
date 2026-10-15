"""The X post image: one bar per benchmark, share of sampled items found broken, with its maker's logo."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LOGOS = ROOT / "post/logos"
BG, INK, MUTED, BASE, BAR, SOFT = "#ffffff", "#141414", "#6b6b6b", "#c8c8c8", "#b8342a", "#f2a08f"
SHORT = {"MBPP (sanitized)": "MBPP", "Terminal-Bench 2.0": "Terminal-Bench 2", "HMMT February 2025": "HMMT 2025"}
MAKER = {  # benchmark -> the organisation that published it
    "mbpp": "google.com", "terminal_bench2": "stanford.edu", "swe_verified": "openai.com", "mmlu_pro": "uwaterloo.ca",
    "humaneval": "openai.com", "simpleqa": "openai.com", "tau2": "sierra.ai", "mmlu": "berkeley.edu",
    "gsm8k": "openai.com", "swe_pro": "scale.com", "hellaswag": "allenai.org", "math500": "openai.com",
    "aime2025": "maa.org", "hmmt2025": "hmmt.org",
}

rows = sorted(json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8")), key=lambda r: -r["rate"])
vals = [100 * r["rate"] for r in rows]

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

w = 0.62
for i, r in enumerate(rows):
    v = vals[i]
    ax.add_patch(FancyBboxPatch((i - w / 2, -2), w, v + 2, boxstyle="round,pad=0,rounding_size=0.05",
                                mutation_aspect=8, fc=BAR, ec="none"))
    ax.text(i, v + 0.4, f"{v:.0f}%" if v >= 1 or v == 0 else f"{v:.1f}%", ha="center", va="bottom",
            color=INK, fontsize=21, fontweight="semibold")

ax.set_xlim(-0.6, len(vals) + 0.6)
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
    # near-white pixels go transparent, so a tilted favicon shows no white corners on the white page
    px = np.array(im)
    px[px[..., :3].min(-1) > 240, 3] = 0
    im = Image.fromarray(px)
    k = box / max(im.size)
    return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)


ANGLE = math.radians(35)
DIR = (math.cos(ANGLE), -math.sin(ANGLE))  # the slant of the labels, in offset points
renderer = fig.canvas.get_renderer()


def width_pt(text, size):
    t = fig.text(0, 0, text, fontsize=size)
    w = t.get_window_extent(renderer).width * 72 / fig.dpi
    t.remove()
    return w


for i, r in enumerate(rows):
    # name first, then the maker's logo, both tilted along one line that runs down to the right
    name, x0, y0 = SHORT.get(r["name"], r["name"]), -2, -14
    ax.annotate(name, (i, 0), xycoords=("data", "axes fraction"), xytext=(x0, y0), textcoords="offset points",
                rotation=-35, rotation_mode="anchor", ha="left", va="center", color=INK, fontsize=18)
    d = width_pt(name, 18) + 24
    im = logo(MAKER[r["bench"]]).rotate(-35, expand=True, resample=Image.BICUBIC)
    ax.add_artist(AnnotationBbox(OffsetImage(im), (i, 0), xycoords=("data", "axes fraction"),
                                 xybox=(x0 + d * DIR[0], y0 + d * DIR[1]), boxcoords="offset points", frameon=False))

fig.text(0.04, 0.9, "What % of each AI benchmark is broken", color=INK, fontsize=36, fontweight="semibold")
fig.text(0.04, 0.03, "Samples of 30 to 164 items each. Audited by Claude, cross-checked by Codex. A lower bound.", color=MUTED, fontsize=14)
fig.text(0.975, 0.03, "enesyilmazcode.github.io/benchbench", color=MUTED, fontsize=14, ha="right")
fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.3)
fig.savefig(ROOT / "post/benchbench-chart-v2.png", facecolor=BG)
print("post/benchbench-chart-v2.png")
