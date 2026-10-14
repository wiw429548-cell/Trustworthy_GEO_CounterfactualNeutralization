"""Pipeline figure for the repository README: the 40 tasks of neutralens/experiments.py grouped by stage.
Icons: Icons8 (Fluency), https://icons8.com."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from style import GEO, ICON, INK, INK2, LEGIT, MANIP, MUTED, OURS, OURS_BG, PANEL, PRES, VIOLET, save

CORE = [  # (icon, title, two description lines, task names, accent)
    ("database", "GEO-bench", ("600 queries, 5 sources", "grouped train / dev / test"), "download · select", OURS),
    ("edit", "Rewrite", ("16 families: 3 legit.,", "7 GEO, 5 manip., 1 fact"), "attack", MANIP),
    ("bot", "Restate", ("fact-only page N(d)", "NLI fact audit"), "neutralize, facts, nli", OURS),
    ("robot-2", "Answer", ("paired contexts, S = 2", "4 engines, 7B to 32B"), "engine_* ×4", INK2),
    ("search", "Score", ("5 detector tasks", "3 answer judges"), "detect_* · judge_*", VIOLET),
    ("combo-chart", "Evaluate", ("dev-calibrated thresholds", "bootstrap · McNemar"), "evaluate · aggregate", PRES),
    ("console", "Paper", ("decomposition, ceiling,", "tables and figures"), "scripts/*.py", PRES),
]
EXT = [  # (icon, title, description, task-name lines, number of tasks)
    ("hacker", "Second attacker", "Phi-3.5-mini rewrites", ["attack_alt", "*_altattack* ×4", "detect_llm_alt"], 6),
    ("synchronize", "Second neutraliser", "Qwen3-8B restatements", ["neutralize_alt", "engine_altneutral", "facts_alt · nli_alt"], 4),
    ("flow-chart", "Research agent", "trajectory hijacking", ["agent_prep_attack", "agent_prep_neutral", "agent_* ×2"], 4),
    ("shield", "Adaptive attacks", "injection · fake facts · evasion", ["adaptive"], 1),
    ("literature", "Literature defences", "paraphrase · spotlight · hierarchy", ["paraphrase", "engine_baselines_* ×2", "judge_baselines_* ×2"], 5),
    ("stopwatch", "Latency and audits", "latency · GEO SI · annotation", ["latency", "judge_si_*", "annotation_export"], 3),
]


def draw():
    W, H = 7.6, 3.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    imgs = {}

    def icon(name, x, y, s):
        if name not in imgs:
            imgs[name] = plt.imread(ICON / f"{name}.png")
        ax.imshow(imgs[name], extent=[x - s / 2, x + s / 2, y - s / 2, y + s / 2], zorder=6, interpolation="lanczos")

    def box(x, y, w, h, fc="white", ec="#d6d4ce", lw=0.7, r=0.06, z=2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    zorder=z, ls=ls))

    def txt(x, y, s, size=6, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        ax.text(x, y, s, fontsize=size, zorder=7, **kw)

    # ------------------------------------------------------------------ core pipeline (17 tasks)
    box(0.05, 1.86, W - 0.10, 1.52, fc=PANEL, ec=PANEL, r=0.08, z=0)
    txt(0.18, 3.25, "Core pipeline", 7.4, ha="left", weight="bold")
    txt(1.18, 3.25, "17 tasks", 6.0, ha="left", color=MUTED)
    n, cw, gap, x0, y0, ch = len(CORE), 0.93, 0.13, 0.17, 1.96, 1.18
    for i, (ic, title, desc, tasks, col) in enumerate(CORE):
        x = x0 + i * (cw + gap)
        box(x, y0, cw, ch, fc="white", ec="#d6d4ce")
        ax.add_patch(Rectangle((x + 0.06, y0 + ch - 0.035), cw - 0.12, 0.035, fc=col, ec="none", zorder=4))
        ax.add_patch(Circle((x + 0.12, y0 + ch - 0.14), 0.065, fc=col, ec="white", lw=0.6, zorder=7))
        txt(x + 0.12, y0 + ch - 0.145, str(i + 1), 5.6, color="white", weight="bold")
        icon(ic, x + cw / 2, y0 + ch - 0.27, 0.30)
        txt(x + cw / 2, y0 + 0.62, title, 7.0, weight="bold")
        txt(x + cw / 2, y0 + 0.455, desc[0], 5.1, color=INK2)
        txt(x + cw / 2, y0 + 0.345, desc[1], 5.1, color=INK2)
        box(x + 0.05, y0 + 0.06, cw - 0.10, 0.17, fc=OURS_BG if col == OURS else "#f1f0ec", ec="none", r=0.03, z=3)
        txt(x + cw / 2, y0 + 0.145, tasks, 4.4, color=INK, family="monospace")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + cw + 0.012, y0 + ch / 2 + 0.05), (x + cw + gap - 0.012, y0 + ch / 2 + 0.05),
                                         arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK2, zorder=5))

    # ------------------------------------------------------------------ ablations and extensions (23 tasks)
    box(0.05, 0.30, W - 0.10, 1.42, fc="white", ec="#c9c6bf", lw=0.8, r=0.08, z=0, ls=(0, (4, 2)))
    txt(0.18, 1.59, "Ablations and extensions", 7.4, ha="left", weight="bold")
    txt(2.02, 1.59, "23 tasks, evaluated by the same step 6", 6.0, ha="left", color=MUTED)
    m, ew, egap, ex0, ey0, eh = len(EXT), 1.13, 0.098, 0.17, 0.42, 1.02
    for i, (ic, title, desc, tasks, k) in enumerate(EXT):
        x = ex0 + i * (ew + egap)
        box(x, ey0, ew, eh, fc="#fbfbfa", ec="#dcdad4")
        icon(ic, x + 0.20, ey0 + eh - 0.22, 0.26)
        txt(x + 0.38, ey0 + eh - 0.17, title, 6.2, ha="left", weight="bold")
        txt(x + 0.38, ey0 + eh - 0.30, f"{k} task" + ("s" if k > 1 else ""), 5.0, ha="left", color=MUTED)
        txt(x + ew / 2, ey0 + 0.565, desc, 5.0, color=INK2)
        box(x + 0.05, ey0 + 0.06, ew - 0.10, 0.385, fc="#f1f0ec", ec="none", r=0.03, z=3)
        top = ey0 + 0.06 + 0.385 / 2 + (len(tasks) - 1) * 0.055
        for j, line in enumerate(tasks):
            txt(x + ew / 2, top - j * 0.11, line, 4.5, color=INK, family="monospace")
    # feed into evaluation
    xe = x0 + 5 * (cw + gap) + cw / 2
    ax.add_patch(FancyArrowPatch((xe, 1.72), (xe, y0 - 0.01), arrowstyle="-|>", mutation_scale=6, lw=0.8,
                                 color=PRES, ls=(0, (3, 2)), zorder=5))
    # footer
    txt(W / 2, 0.15, "run_all.py schedules all 40 tasks over the visible GPUs; every task writes a done-file and caches "
                     "its LLM outputs, so re-running the command resumes.", 5.6, color=MUTED, style="italic")
    save(fig, "pipeline")


if __name__ == "__main__":
    draw()
