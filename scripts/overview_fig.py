"""Figure 1: the counterfactual-neutralization pipeline on a real test example (GEO-bench query q439, citation-hook
rewrite, Qwen2.5-7B engine). Every text snippet and number is read from data/ and results/ at drawing time.
Icons: Icons8 (Fluency), https://icons8.com."""
import hashlib
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from style import (DATA, GEO, GRID, ICON, INK, INK2, LEGIT, MANIP, MANIP_BG, MUTED, OURS, OURS_BG, PANEL, PRES,
                   PRES_BG, RES, save)

QID, FAM, ENGINE = "q439", "citation_hook", "Qwen_Qwen2_5_7B_Instruct"


def _h(t):
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


def _rj(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def example():
    """texts and visibility shares of the example from the raw files"""
    q = next(r for r in _rj(DATA / "geo" / "queries.jsonl") if r["qid"] == QID)
    var = next(v for v in _rj(DATA / "geo" / "variants.jsonl") if v["qid"] == QID and v["family"] == FAM)
    neutral = {}
    for r in _rj(DATA / "geo" / "neutral.jsonl"):
        neutral[r["hash"]] = r["neutral"]
    sh, ans = defaultdict(list), {}
    with open(DATA / "engine" / ENGINE / "gens.jsonl", encoding="utf-8") as f:
        for l in f:
            if f'"qid": "{QID}"' not in l:
                continue
            r = json.loads(l)
            if r["cond"] in ("benign", FAM) and r["ctx"] in ("plain", "neutral_target"):
                sh[(r["cond"], r["ctx"])].append(r["shares"][q["target"]])
                ans[(r["cond"], r["ctx"], r["sample"])] = (r["answer"], r["order"])
    m = {k: sum(v) / len(v) for k, v in sh.items()}
    return {"query": q["query"], "orig": q["sources"][q["target"]], "rewrite": var["text"],
            "neutral": neutral[_h(var["text"])], "V_plain": m[(FAM, "plain")], "V_neut": m[(FAM, "neutral_target")],
            "M_orig": m[("benign", "plain")] - m[("benign", "neutral_target")], "S": len(sh[(FAM, "plain")]),
            "answer": ans[(FAM, "plain", 1)][0], "order": ans[(FAM, "plain", 1)][1], "target": q["target"]}


def draw():
    ex = example()
    A = json.load(open(RES / "extra" / "analysis.json", encoding="utf-8"))["Qwen/Qwen2.5-7B-Instruct"]
    man = A["group"]["manipulative"]
    pres_all = json.load(open(RES / "preservation.json", encoding="utf-8"))["by_family"]
    n_all = sum(v["n"] for v in pres_all.values())
    rec = sum(v["fact_recall"] * v["n"] for v in pres_all.values()) / n_all
    prec = sum(v["fact_precision"] * v["n"] for v in pres_all.values()) / n_all

    W, H = 7.0, 3.02
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    imgs = {}

    def icon(name, x, y, s):
        if name not in imgs:
            imgs[name] = plt.imread(ICON / f"{name}.png")
        ax.imshow(imgs[name], extent=[x - s / 2, x + s / 2, y - s / 2, y + s / 2], zorder=6, interpolation="lanczos")

    def box(x, y, w, h, fc="white", ec="#cfcdc7", lw=0.6, r=0.045, z=1, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    zorder=z, ls=ls))

    def arrow(p, q, col=INK2, lw=0.75, rad=0.0, ms=6.5, z=4, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))

    def txt(x, y, s, size=6.0, **kw):
        kw.setdefault("ha", "left")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        return ax.text(x, y, s, fontsize=size, zorder=7, **kw)

    def width(t):
        return t.get_window_extent(renderer=R).width / fig.dpi

    def segs(x, y, parts, size=5.5):
        """one line made of (text, colour, background or None) segments"""
        for s, col, bg in parts:
            kw = {"bbox": dict(fc=bg, ec="none", pad=0.35, boxstyle="round,pad=0.12")} if bg else {}
            t = txt(x, y, s, size, color=col, **kw)
            x += width(t)
        return x

    def badge(x, y, n, col):
        ax.add_patch(Circle((x, y), 0.072, fc=col, ec="white", lw=0.8, zorder=7))
        txt(x, y - 0.003, str(n), 6.0, ha="center", color="white", weight="bold")

    # ------------------------------------------------------------------ query strip
    box(0.02, 2.74, 6.96, 0.24, fc=PANEL, ec=PANEL, r=0.05)
    icon("search", 0.15, 2.86, 0.15)
    txt(0.27, 2.86, r"query $q$:", 6.2, color=INK2)
    txt(0.62, 2.86, f"“{ex['query']}”", 6.4, style="italic")
    txt(3.45, 2.86, r"5 retrieved sources $D=(d_1,\dots,d_5)$; the publisher controls the target $d_t$ (GEO-bench)",
        5.8, color=INK2)

    panels = [(0.02, 1.70, "Rewrite", MANIP), (1.80, 1.70, "Fact-only restatement", OURS),
              (3.58, 1.80, "Paired generation", INK2), (5.46, 1.52, "Decompose and act", PRES)]
    for i, (x0, w, title, col) in enumerate(panels):
        box(x0, 0.06, w, 2.60, fc="white", ec="#dcdad4", lw=0.7, r=0.06, z=0)
        badge(x0 + 0.13, 2.52, i + 1, col)
        txt(x0 + 0.25, 2.515, title, 7.0, weight="bold")
    for xa in (1.73, 3.51, 5.39):
        ax.add_patch(FancyArrowPatch((xa, 1.36), (xa + 0.085, 1.36), arrowstyle="simple,head_width=4,head_length=3",
                                     mutation_scale=1.6, fc="#b9b6ae", ec="none", zorder=8))

    # ------------------------------------------------------------------ (1) rewrite
    x0 = 0.02
    for k in range(5):
        xx = x0 + 0.14 + k * 0.30
        tgt = k == 1
        box(xx, 2.16, 0.24, 0.20, fc=MANIP_BG if tgt else "#efeee9", ec=MANIP if tgt else "#cfcdc7",
            lw=0.8 if tgt else 0.5, r=0.03)
        txt(xx + 0.12, 2.26, r"$d_t$" if tgt else f"$d_{k + 1}$", 5.6, ha="center", color=MANIP if tgt else MUTED)
    box(x0 + 0.09, 1.66, 1.52, 0.42, fc="#fbfbfa", ec="#d9d7d1", r=0.035)
    txt(x0 + 0.14, 1.98, r"original $d_t$", 5.6, weight="bold", color=INK2)
    txt(x0 + 0.14, 1.80, "The causes of World War II have been given\nconsiderable attention by historians. The ...",
        5.2, color=INK2, linespacing=1.15)
    arrow((x0 + 0.22, 1.655), (x0 + 0.22, 1.47), lw=0.7)
    icon("edit", x0 + 0.40, 1.565, 0.14)
    txt(x0 + 0.51, 1.565, r"attacker LLM applies family $f$", 5.4, color=INK2)
    chips = [[("legitimate", 3, LEGIT), ("GEO", 7, GEO)], [("manipulative", 5, MANIP), ("false fact", 1, MUTED)]]
    for row, cy in zip(chips, (1.40, 1.26)):
        cx = x0 + 0.12
        for name, n, col in row:
            t = txt(cx + 0.05, cy, f"{name} ×{n}", 5.0, color=INK)
            wd = width(t)
            box(cx, cy - 0.055, wd + 0.10, 0.11, fc="white", ec=col, lw=0.8, r=0.03, z=5)
            cx += wd + 0.16
    txt(x0 + 1.10, 1.40, "16 families", 5.0, color=MUTED, style="italic")
    # rewritten page with highlighted presentation span
    box(x0 + 0.09, 0.14, 1.52, 1.03, fc="white", ec=MANIP, lw=0.9, r=0.04)
    txt(x0 + 0.14, 1.075, r"rewrite $\tilde d$ (citation hook)", 5.6, weight="bold", color=MANIP)
    y, dy = 0.935, 0.118
    segs(x0 + 0.14, y, [("The definitive answer to the main causes of", INK, PRES_BG)], size=5.3)
    segs(x0 + 0.14, y - dy, [("World War II is that", INK, PRES_BG), (" they include the political", INK, None)],
         size=5.3)
    segs(x0 + 0.14, y - 2 * dy, [("takeover of Germany by Adolf Hitler and the", INK, None)], size=5.3)
    segs(x0 + 0.14, y - 3 * dy, [("Nazi Party in 1933, Japanese militarism", INK, None)], size=5.3)
    segs(x0 + 0.14, y - 4 * dy, [("against China, Italian aggression against", INK, None)], size=5.3)
    segs(x0 + 0.14, y - 5 * dy, [("Ethiopia, ... significantly contributed ...", INK, None)], size=5.3)
    ax.add_patch(Rectangle((x0 + 0.15, 0.19), 0.09, 0.055, fc=PRES_BG, ec="none", zorder=6))
    txt(x0 + 0.27, 0.218, "presentation", 5.0, color=INK2)
    ax.add_patch(Rectangle((x0 + 0.78, 0.19), 0.09, 0.055, fc="white", ec="#bdbab3", lw=0.4, zorder=6))
    txt(x0 + 0.90, 0.218, "factual claims", 5.0, color=INK2)

    # ------------------------------------------------------------------ (2) neutraliser
    x0 = 1.80
    icon("bot", x0 + 0.24, 2.22, 0.30)
    txt(x0 + 0.43, 2.29, r"neutraliser $\mathcal{N}$ (LLM)", 5.9, weight="bold")
    txt(x0 + 0.43, 2.14, "keeps every claim, true or false;\ndrops self-description, endorsements,\n"
        "urgency, instructions to AI systems", 5.0, color=INK2, va="top", linespacing=1.15)
    arrow((x0 + 0.24, 2.05), (x0 + 0.24, 1.86), lw=0.7)
    box(x0 + 0.09, 0.56, 1.52, 1.28, fc=OURS_BG, ec=OURS, lw=0.9, r=0.04)
    icon("list", x0 + 0.20, 1.74, 0.12)
    txt(x0 + 0.30, 1.74, r"$\mathcal{N}(\tilde d)$: facts stated by the page", 5.6, weight="bold", color=OURS)
    facts = [l[2:].strip() for l in ex["neutral"].split("\n") if l.startswith("- ")]
    shown = facts[:4]
    yy = 1.60
    for fct in shown:
        words, line, lines = fct.split(), "", []
        for w_ in words:
            if len(line) + len(w_) + 1 > 41:
                lines.append(line)
                line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:2]):
            txt(x0 + 0.15 + (0.0 if j == 0 else 0.06), yy, ("• " if j == 0 else "") + ln, 5.0, color=INK)
            yy -= 0.098
        yy -= 0.012
    txt(x0 + 0.15, yy, f"… {len(facts)} facts in total; the opening claim of being", 5.0, color=INK2,
        style="italic")
    txt(x0 + 0.15, yy - 0.10, "“the definitive answer” is gone", 5.0, color=INK2, style="italic")
    # audit badge
    box(x0 + 0.09, 0.14, 1.52, 0.36, fc="#fbfbfa", ec="#d9d7d1", r=0.035)
    icon("checked-checkbox", x0 + 0.21, 0.32, 0.15)
    txt(x0 + 0.33, 0.41, "fact audit (atomic facts + NLI)", 5.4, weight="bold", color=INK2)
    txt(x0 + 0.33, 0.295, f"recall {rec:.2f}: facts of the page kept", 5.1, color=INK2)
    txt(x0 + 0.33, 0.20, f"precision {prec:.2f}: nothing added", 5.1, color=INK2)

    # ------------------------------------------------------------------ (3) paired generation
    x0 = 3.58
    txt(x0 + 0.12, 2.30, r"same sources, source order $\pi_s$ and seed $s$ in both contexts", 4.95, color=INK2)
    rows = [(1.98, r"$D$", r"$\tilde d$", MANIP, MANIP_BG, ex["V_plain"]),
            (1.38, r"$D^{(t)}$", r"$\mathcal{N}(\tilde d)$", OURS, OURS_BG, ex["V_neut"])]
    pos_t = ex["order"].index(ex["target"])
    for y, lab, tl, col, bg, v in rows:
        txt(x0 + 0.10, y, lab, 6.4)
        for k in range(5):
            xx = x0 + 0.36 + k * 0.205
            if k == pos_t:
                box(xx, y - 0.10, 0.18, 0.20, fc=bg, ec=col, lw=0.8, r=0.03)
                txt(xx + 0.09, y, tl, 5.0, ha="center", color=col)
            else:
                box(xx, y - 0.10, 0.18, 0.20, fc="#efeee9", ec="#cfcdc7", lw=0.45, r=0.03)
                txt(xx + 0.09, y, f"[{k + 1}]", 4.8, ha="center", color=MUTED)
        arrow((x0 + 1.39, y), (x0 + 1.47, y), lw=0.6)
        icon("robot-2", x0 + 1.56, y, 0.20)
        # visibility bar
        bx0, bw_ = x0 + 0.36, 1.02
        ax.add_patch(Rectangle((bx0, y - 0.215), bw_, 0.055, fc="#efeee9", ec="none", zorder=3))
        ax.add_patch(Rectangle((bx0, y - 0.215), bw_ * v / 0.5, 0.055, fc=col, ec="none", zorder=4))
        txt(bx0 + bw_ + 0.05, y - 0.19, f"$V_t={v:.2f}$", 5.4, color=col, weight="bold")
    txt(x0 + 0.12, 1.06, r"engine $G$ answers with inline citations; $V_t$ = share of", 5.0, color=INK2)
    txt(x0 + 0.12, 0.965, r"position-weighted cited words (PAWC), mean of $S=" + str(ex["S"]) + r"$", 5.0,
        color=INK2)
    # answer excerpt
    box(x0 + 0.09, 0.42, 1.62, 0.45, fc="#fbfbfa", ec="#d9d7d1", r=0.035)
    txt(x0 + 0.14, 0.79, r"answer from $D$, sample 2 (excerpt)", 5.0, color=INK2, weight="bold")
    segs(x0 + 0.14, 0.665, [("The main causes of World War II include the political", INK, None)], size=5.0)
    segs(x0 + 0.14, 0.56, [("takeover of Germany by Adolf Hitler and the Nazi Party", INK, None)], size=5.0)
    segs(x0 + 0.14, 0.455, [("in 1933 ", INK, None), ("[1]", MANIP, MANIP_BG), ("[3][5], Japanese militarism ...", INK, None)],
         size=5.0)
    M_rw = ex["V_plain"] - ex["V_neut"]
    box(x0 + 0.09, 0.14, 1.62, 0.22, fc=PANEL, ec=PANEL, r=0.035)
    txt(x0 + 0.14, 0.25, rf"gap $M(\tilde d)={ex['V_plain']:.2f}-{ex['V_neut']:.2f}={M_rw:.2f}$", 5.8, weight="bold")
    txt(x0 + 1.12, 0.25, rf"original: $M(d_t)={ex['M_orig']:+.2f}$", 5.0, color=INK2)

    # ------------------------------------------------------------------ (4) decompose and act
    x0 = 5.46
    G, F, P = man["G"][0], man["F"][0], man["P"][0]
    txt(x0 + 0.12, 2.33, r"$\Delta = F + P$, mean over manipulative", 5.3, color=INK2)
    txt(x0 + 0.12, 2.225, "rewrites (Qwen2.5-7B, 299 test queries)", 5.3, color=INK2)
    sc = 0.66 / G
    bx, by, bw = x0 + 0.16, 1.44, 0.24
    ax.add_patch(Rectangle((bx, by), bw, F * sc, fc=OURS, ec="white", lw=0.8, zorder=4))
    ax.add_patch(Rectangle((bx, by + F * sc), bw, P * sc, fc=PRES, ec="white", lw=0.8, zorder=4))
    ax.plot([bx - 0.05, bx + bw + 0.05], [by, by], color=INK2, lw=0.6, zorder=5)
    txt(bx + bw + 0.06, by + F * sc / 2, rf"fact-borne $F={F:.3f}$", 5.2, color=OURS, weight="bold")
    txt(bx + bw + 0.06, by + F * sc + P * sc / 2 + 0.06, rf"presentation $P={P:.3f}$", 5.2, color=PRES,
        weight="bold")
    txt(bx + bw + 0.06, by + F * sc + P * sc / 2 - 0.06, rf"$=M(\tilde d)-M(d_t)$, {100 * P / G:.0f}% of $\Delta$",
        5.0, color=PRES)
    uses = [("combo-chart", "Audit", "presentation share per family"),
            ("search", "Detect", r"flag $d$ if $M(d)>\tau$"),
            ("replace", "Replace / All", r"serve $\mathcal{N}(d)$: facts stay"),
            ("robot-2", "Agents", "planning gap on next queries")]
    yy = 1.33
    for ic, head, body in uses:
        box(x0 + 0.08, yy - 0.265, 1.36, 0.265, fc="white", ec="#d9d7d1", lw=0.55, r=0.03)
        icon(ic, x0 + 0.20, yy - 0.1325, 0.15)
        txt(x0 + 0.32, yy - 0.08, head, 5.6, weight="bold")
        txt(x0 + 0.32, yy - 0.19, body, 4.9, color=INK2)
        yy -= 0.30
    save(fig, "overview")


if __name__ == "__main__":
    draw()
