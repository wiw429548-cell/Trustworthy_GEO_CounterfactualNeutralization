"""Every figure of the paper, drawn with matplotlib from ../results (no number is typed by hand).

overview.pdf       schematic of counterfactual neutralization (icons: Icons8, Fluency style, https://icons8.com)
decomposition.pdf  fact-borne vs presentation-borne visibility gain per rewrite family + share across engines
reliability.pdf    split-half reliability of the per-document gap M and overlap of M across groups
lofo.pdf           leave-one-family-out AUROC of the influence gap vs text detectors
tradeoff.pdf       ASR on manipulative rewrites vs retained legitimate gain for every defence
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "assets" / "figures"
ICON = ROOT / "assets" / "icons"
FONT_DIR = Path("C:/Users/Administrator/AppData/Local/Programs/MiKTeX/fonts/opentype/public/libertine")

# ---------------------------------------------------------------- style (ACM: Linux Biolinum for sans text)
fam = "Arial" if any(x.name == "Arial" for x in fm.fontManager.ttflist) else "DejaVu Sans"
for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf"):
    if (FONT_DIR / f).exists():
        fm.fontManager.addfont(str(FONT_DIR / f))
        fam = "Linux Biolinum O"
plt.rcParams.update({
    "font.family": fam, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#1f2328", "text.color": "#1f2328",
    "legend.frameon": False, "savefig.dpi": 300})
if fam not in ("Arial", "DejaVu Sans"):
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic",
                         "mathtext.bf": fam + ":bold", "mathtext.cal": "cmsy10"})
# reference categorical palette (validated order) + text inks
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8a8984", "#e4e3df"
GROUP_COL = {"legit": C["aqua"], "geo": C["yellow"], "manipulative": C["red"], "fact": MUTED}

CFG_FAM = {"legit": ["clarity", "fluency", "structure"],
           "geo": ["authoritative", "technical_terms", "unique_words", "keyword_stuffing", "statistics", "quotation",
                   "cite_sources"],
           "manipulative": ["hidden_instruction", "comparative_disparagement", "fake_endorsement", "citation_hook",
                            "emotional_urgency"],
           "fact": ["fact_poison"]}
NICE = {"clarity": "clarity", "fluency": "fluency", "structure": "structure", "authoritative": "authoritative",
        "technical_terms": "technical terms", "unique_words": "unique words", "keyword_stuffing": "keyword stuffing",
        "statistics": "statistics", "quotation": "quotation", "cite_sources": "cite sources",
        "hidden_instruction": "AI-directed instr.", "comparative_disparagement": "disparagement",
        "fake_endorsement": "fake endorsement", "citation_hook": "citation hook", "emotional_urgency": "urgency",
        "fact_poison": "false fact"}
ENG = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen/Qwen3-8B": "Qwen3-8B",
       "ibm-granite/granite-3.1-8b-instruct": "Granite-3.1-8B", "Qwen/Qwen2.5-32B-Instruct-AWQ": "Qwen2.5-32B"}
MAIN = "Qwen_Qwen2_5_7B_Instruct"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ================================================================ Figure 1: overview schematic
def overview():
    from overview_fig import draw
    draw()


# ================================================================ Figure 2: gain decomposition
def decomposition():
    A = load(RES / "extra" / "analysis.json")
    main = A["Qwen/Qwen2.5-7B-Instruct"]["family"]
    fig = plt.figure(figsize=(7.0, 2.05))
    ax = fig.add_axes([0.055, 0.30, 0.62, 0.62])
    order = [f for g in ("legit", "geo", "manipulative", "fact") for f in CFG_FAM[g]]
    xs, x, pos = [], 0.0, {}
    for g in ("legit", "geo", "manipulative", "fact"):
        for f in CFG_FAM[g]:
            pos[f] = x
            x += 1
        x += 0.6
    for f in order:
        G, F, P = main[f]["G"], main[f]["F"], main[f]["P"]
        xp = pos[f]
        fb = F[0]
        ax.bar(xp, fb, 0.72, color=C["blue"], zorder=3, linewidth=0)
        base = max(fb, 0)
        ax.bar(xp, P[0], 0.72, bottom=base, color=C["orange"], zorder=3, edgecolor="white",
               linewidth=0.8 if fb > 0 else 0)
        ax.errorbar(xp, G[0], yerr=[[G[0] - G[1]], [G[2] - G[0]]], fmt="D", ms=2.4, color=INK, mec=INK,
                    elinewidth=0.6, capsize=1.3, zorder=5)
    ax.axhline(0, color=INK2, lw=0.6, zorder=4)
    ax.set_xticks([pos[f] for f in order])
    ax.set_xticklabels([NICE[f] for f in order], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("target visibility gain\n(PAWC share)")
    ax.set_xlim(-0.6, x - 0.9)
    for g in ("legit", "geo", "manipulative", "fact"):
        fs = CFG_FAM[g]
        ax.axvspan(pos[fs[0]] - 0.5, pos[fs[-1]] + 0.5, color=GROUP_COL[g], alpha=0.06, lw=0, zorder=0)
    grid(ax)
    ax.set_ylim(top=0.245)
    top = ax.get_ylim()[1]
    for g, lab in (("legit", "legitimate"), ("geo", "GEO"), ("manipulative", "manipulative"), ("fact", "fact")):
        fs = CFG_FAM[g]
        a, b = pos[fs[0]], pos[fs[-1]]
        ax.plot([a - 0.35, b + 0.35], [top * 1.02] * 2, color=GROUP_COL[g], lw=2.0, solid_capstyle="butt",
                clip_on=False)
        ax.text((a + b) / 2, top * 1.07, lab, ha="center", va="bottom", fontsize=6.8, color=INK, clip_on=False)
    ax.set_ylim(min(-0.02, ax.get_ylim()[0]), top)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="fact-borne (survives $\\mathcal{N}$)"),
                       Patch(color=C["orange"], label="presentation-borne"),
                       Line2D([], [], marker="D", ls="", color=INK, ms=2.6, label="total gain, 95% CI")],
              loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), handlelength=1.1, borderaxespad=0.1,
              columnspacing=1.2)
    ax.text(-0.06, 1.13, "(a)", transform=ax.transAxes, fontsize=8, weight="bold")
    # ------------------------------------------------------------ (b) presentation share by group x engine
    bx = fig.add_axes([0.765, 0.30, 0.225, 0.62])
    groups = ["legit", "geo", "manipulative"]
    engs = list(ENG)
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"]]
    marks = ["o", "s", "^", "D"]
    for i, e in enumerate(engs):
        if e not in A:
            continue
        for j, g in enumerate(groups):
            s = A[e]["group"][g]["share_P"]
            y = j + (i - 1.5) * 0.17
            bx.errorbar(min(s[0], 1.05), y, xerr=[[s[0] - s[1]], [min(s[2], 1.1) - s[0]]], fmt=marks[i], ms=3.0,
                        color=cols[i], mec="white", mew=0.4, elinewidth=0.6, capsize=0, zorder=4,
                        label=ENG[e] if j == 0 else None)
    bx.set_yticks(range(3))
    bx.set_yticklabels(["legitimate", "GEO", "manipulative"])
    bx.invert_yaxis()
    bx.set_xlim(0, 1.1)
    bx.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    bx.set_xlabel("presentation-borne share of gain")
    grid(bx, "x")
    bx.legend(loc="upper center", bbox_to_anchor=(0.45, -0.30), ncol=2, handletextpad=0.2, columnspacing=0.6,
              borderaxespad=0)
    bx.text(-0.42, 1.13, "(b)", transform=bx.transAxes, fontsize=8, weight="bold")
    save(fig, "decomposition")


# ================================================================ Figure 3: reliability of M
def reliability():
    A = load(RES / "extra" / "analysis.json")
    docs = load(RES / "extra" / "m_docs_main.json")
    m0 = np.array([d["m0"] for d in docs])
    m1 = np.array([d["m1"] for d in docs])
    fig, (b, a) = plt.subplots(1, 2, figsize=(3.35, 1.6), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1.05]})
    mbar = (m0 + m1) / 2
    grp = np.array([d["group"] for d in docs])
    bins = np.linspace(-0.5, 0.5, 41)
    lab = {"benign": "original", "legit": "legitimate", "geo": "GEO", "manipulative": "manipulative"}
    col = {"benign": INK2, "legit": GROUP_COL["legit"], "geo": GROUP_COL["geo"], "manipulative": GROUP_COL["manipulative"]}
    for g in ("benign", "legit", "geo", "manipulative"):
        v = mbar[grp == g]
        hist, edges = np.histogram(v, bins=bins, density=True)
        ctr = (edges[:-1] + edges[1:]) / 2
        b.plot(ctr, hist, color=col[g], lw=1.0 if g != "benign" else 1.2, ls="-" if g != "benign" else "--",
               label=lab[g])
    b.set_xlim(-0.5, 0.5)
    b.set_xticks([-0.4, 0, 0.4])
    b.set_xlabel("gap $M$ (Qwen2.5-7B)")
    b.set_ylabel("density")
    b.set_ylim(bottom=0)
    b.legend(loc="upper right", bbox_to_anchor=(1.13, 1.07), handlelength=0.9, handletextpad=0.3, labelspacing=0.12, fontsize=5.6)
    b.text(-0.40, 1.04, "(a)", transform=b.transAxes, fontsize=8, weight="bold")
    # (b) AUROC (manipulative vs original) predicted by the Gaussian model vs number of samples S
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"]]
    marks = ["o", "s", "^", "D"]
    Ss = [1, 2, 4, 8, 16]
    for i, e in enumerate(ENG):
        if e not in A:
            continue
        c = A[e]["ceiling"]["manipulative"]
        a.plot(range(len(Ss)), [c["pred"][str(S)] for S in Ss], color=cols[i], lw=1.0, label=ENG[e])
        a.plot([len(Ss) + 0.4], [c["pred"]["inf"]], marker="_", ms=7, mew=1.4, color=cols[i])
        a.scatter([0, 1], [c["emp_S1"], c["emp_S2"]], s=12, marker=marks[i], color=cols[i], edgecolor="white",
                  linewidth=0.4, zorder=4)
    a.set_xticks(list(range(len(Ss))) + [len(Ss) + 0.4])
    a.set_xticklabels([str(S) for S in Ss] + [r"$\infty$"])
    a.set_xlabel("samples $S$ per context")
    a.set_ylabel("AUROC, manip. vs original")
    a.set_ylim(0.44, 0.75)
    a.axvspan(-0.3, 1.3, color="#f4f4f2", zorder=0)
    a.text(0.5, 0.735, "measured", ha="center", fontsize=5.6, color=INK2)
    grid(a)
    a.legend(loc="lower right", bbox_to_anchor=(1.04, -0.03), handlelength=1.0, labelspacing=0.1, fontsize=5.3)
    a.text(-0.42, 1.04, "(b)", transform=a.transAxes, fontsize=8, weight="bold")
    save(fig, "reliability")


# ================================================================ Figure 4: leave-one-family-out transfer
def lofo():
    E = load(RES / "eval" / f"{MAIN}.json")["lofo"]["families"]
    fams = [f for f in CFG_FAM["geo"] + CFG_FAM["manipulative"] if f in E]
    series = [("cgeo_lofo", "C-GEO-Guard-lite (re-trained w/o family)", C["orange"], "s"),
              ("sci", "SCI-lite (LLM judge)", C["violet"], "^"),
              ("M", "counterfactual gap $M$ (no training)", C["blue"], "o"),
              ("amort_lofo", "amortised gap (re-trained w/o family)", C["aqua"], "D")]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ys = np.arange(len(fams))
    for k, (key, lab, c, m) in enumerate(series):
        v = [E[f][key]["AUROC"] for f in fams]
        ax.scatter(v, ys + (k - 1.5) * 0.12, s=13, marker=m, color=c, edgecolor="white", linewidth=0.4, zorder=4,
                   label=lab)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.4, zorder=0)
    ng_ = len([f for f in fams if f in CFG_FAM["geo"]])
    ax.axhspan(-0.5, ng_ - 0.5, color=GROUP_COL["geo"], alpha=0.06, lw=0, zorder=0)
    ax.axhspan(ng_ - 0.5, len(fams) - 0.5, color=GROUP_COL["manipulative"], alpha=0.06, lw=0, zorder=0)
    ax.axvline(0.5, color=INK2, lw=0.6, ls=":", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([NICE[f] for f in fams])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("AUROC of held-out family vs. original pages")
    # group brackets
    ng = len([f for f in fams if f in CFG_FAM["geo"]])
    for (a, b, g) in ((0, ng - 1, "geo"), (ng, len(fams) - 1, "manipulative")):
        ax.plot([1.005, 1.005], [a - 0.35, b + 0.35], color=GROUP_COL[g], lw=2.0, clip_on=False,
                solid_capstyle="butt", transform=ax.get_yaxis_transform())
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.18), ncol=2, handletextpad=0.1, columnspacing=0.5,
              borderaxespad=0, fontsize=6.0)
    save(fig, "lofo")


# ================================================================ Figure 5: safety / fairness trade-off
def tradeoff():
    E = load(RES / "eval" / f"{MAIN}.json")["baselines_subset"]["defenses"]
    names = {"none": "no defence", "paraphrase_all": "paraphrase all", "neutral_all": "neutralise all",
             "filter:ppl": "perplexity", "filter:sci": "SCI-lite", "filter:guard": "guard LLM",
             "filter:cgeo": "C-GEO-lite", "filter:amort_all": "amortised gap", "filter:M_fixed": r"gap $M>\tau$",
             "replace:M_fixed": r"gap $M>\tau$", "replace:M": "gap, FPR-cal.", "replace:amort_all": "amortised",
             "replace:cgeo": "C-GEO-lite", "shield": None, "ignore_instr": None, "spotlight": None,
             "filter:pi": None, "replace:pi": None}
    kind = lambda k: ("filter" if k.startswith("filter:") else "replace" if k.startswith("replace:")
                      else "transform" if k in ("neutral_all", "paraphrase_all") else "prompt")
    style = {"prompt": (MUTED, "o", "none / prompt-level"), "transform": (C["aqua"], "D", "rewrite every source"),
             "filter": (C["orange"], "s", "remove flagged page"), "replace": (C["blue"], "^", r"replace flagged page by $\mathcal{N}(d)$")}
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    floor = E["none"]["ASR_fact"]
    ax.axhline(floor, color=INK2, lw=0.6, ls="--", zorder=1)
    ax.axhspan(0.15, floor, color="#efeee9", alpha=0.7, lw=0, zorder=0)
    ax.annotate("", xy=(0.42, 0.172), xytext=(0.12, 0.222), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7),
                zorder=2)
    ax.text(0.44, 0.166, "better", fontsize=5.6, color=MUTED, style="italic")
    ax.text(0.0, floor + 0.004, "noise floor (false-fact rewrite, zero mean gain)", fontsize=5.6, color=INK2,
            va="bottom")
    offs = {"none": (-30, 7), "paraphrase_all": (-12, -8), "neutral_all": (5, 0), "filter:ppl": (5, 0),
            "filter:sci": (-27, 0), "filter:guard": (-14, 8), "filter:cgeo": (-37, 0), "filter:amort_all": (-52, 0),
            "filter:M_fixed": (5, 0), "replace:M_fixed": (5, -7), "replace:M": (-46, 2),
            "replace:amort_all": (-8, 8), "replace:cgeo": (-38, -4)}
    for kd in ("prompt", "transform", "filter", "replace"):
        c, m, ll = style[kd]
        first = True
        for k, lab in names.items():
            if kind(k) != kd or k not in E or E[k].get("legit_gain_retention") is None:
                continue
            xv, yv = E[k]["legit_gain_retention"], E[k]["ASR_manipulative"]
            ax.scatter(xv, yv, s=17, marker=m, color=c, edgecolor="white", linewidth=0.4, zorder=4,
                       label=ll if first else None)
            first = False
            if lab:
                ax.annotate(lab, (xv, yv), xytext=offs.get(k, (4, 0)), textcoords="offset points", fontsize=5.5,
                            color=INK2, va="center")
    ax.set_xlabel("legitimate gain kept (1 = all of it)")
    ax.set_ylabel("ASR on manipulative rewrites")
    ax.set_xlim(-0.06, 1.17)
    ax.set_ylim(0.15, 0.53)
    grid(ax, "both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8,
              labelspacing=0.2, fontsize=5.9, borderaxespad=0.2)
    save(fig, "tradeoff")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    overview()
    decomposition()
    reliability()
    lofo()
    tradeoff()
    print("figures ->", FIG)
