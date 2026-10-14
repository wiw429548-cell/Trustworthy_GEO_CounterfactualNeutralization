"""Paper tables (results/paper_tables/*.tex), generated from ../results only."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "results" / "paper_tables"
ENGS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen_Qwen2_5_7B_Instruct", "Qwen2.5-7B"),
        ("Qwen/Qwen3-8B", "Qwen_Qwen3_8B", "Qwen3-8B"),
        ("ibm-granite/granite-3.1-8b-instruct", "ibm_granite_granite_3_1_8b_instruct", "Granite-3.1-8B"),
        ("Qwen/Qwen2.5-32B-Instruct-AWQ", "Qwen_Qwen2_5_32B_Instruct_AWQ", "Qwen2.5-32B$^\\dagger$")]
MAIN = "Qwen_Qwen2_5_7B_Instruct"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f2(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}"


def pval(p):
    if p is None:
        return "--"
    if p < 1e-3:
        e = f"{p:.0e}".split("e")
        return f"${e[0]}{{\\times}}10^{{{int(e[1])}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _num(cell):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cell.replace("$", ""))
    return float(m.group(0)) if m and "times" not in cell else None


def stylize(s, best=None, ours=()):
    """grey header, zebra body rows, light-blue rows for our method, best value per column in bold blue.
    best: {column index: 'max' | 'min' | 'all'} over body rows."""
    lines = s.split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip() == r"\toprule")
    i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].strip() == r"\midrule")
    i2 = next(i for i, l in enumerate(lines) if l.strip() == r"\bottomrule")
    for i in range(i0 + 1, i1):
        if lines[i].rstrip().endswith(r"\\"):
            lines[i] = r"\rowcolor{tblhead}" + lines[i]
    body = [i for i in range(i1 + 1, i2) if lines[i].rstrip().endswith(r"\\")]
    if best:
        cells = {i: [c.strip() for c in lines[i].rstrip()[:-2].split(" & ")] for i in body}
        for col, mode in best.items():
            vals = {i: _num(c[col]) for i, c in cells.items() if col < len(c) and _num(c[col]) is not None}
            if not vals:
                continue
            target = None if mode == "all" else (max(vals.values()) if mode == "max" else min(vals.values()))
            for i, v in vals.items():
                if mode == "all" or abs(v - target) < 1e-9:
                    c = cells[i][col]
                    head, sep, tail = c.partition(" [")
                    cells[i][col] = r"\best{" + head + "}" + (sep + tail if sep else "")
        for i in body:
            lines[i] = " & ".join(cells[i]) + r" \\"
    k = 0
    for i in body:
        if any(o in lines[i] for o in ours):
            lines[i] = r"\rowcolor{oursbg}" + lines[i]
        else:
            lines[i] = (r"\rowcolor{tblrow}" if k % 2 else "") + lines[i]
        k += 1
    return "\n".join(lines)


def write(name, s, best=None, ours=()):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(stylize(s, best, ours), encoding="utf-8")


# ---------------------------------------------------------------------------- Table: four engines at a glance
def engines():
    A = load(RES / "extra" / "analysis.json")
    rows = []
    for full, sl, nice in ENGS:
        E = load(RES / "eval" / f"{sl}.json")
        a = A[full]
        sh = lambda g: a["group"][g]["share_P"][0]
        det = E["detection"]["M"]
        none = E["defenses"]["none"]["ASR_manipulative"]
        rep = E["defenses"]["replace:M_fixed"]["ASR_manipulative"]
        pr = E["asr_paired_vs_neutralens"]["none"]
        rows.append(f"{nice} & {E['n_test_queries']} & {f2(a['benign_target_visibility'])} & "
                    f"{pct(sh('legit'))} & {pct(sh('geo'))} & {pct(sh('manipulative'))} & "
                    f"{f2(a['reliability']['split_half_r'])} & {f2(det['AUROC[manipulative vs benign]'])} & "
                    f"{f2(a['ceiling']['manipulative']['pred']['inf'])} & {f2(det['AUROC[legit vs benign]'])} & "
                    f"{f2(a['ceiling']['legit']['pred']['inf'])} & {f2(none)} & {f2(rep)} & {pval(pr['p_mcnemar'])} \\\\")
    s = r"""\begin{table*}[t]
\caption{Counterfactual neutralization on four engines (GEO-bench test queries, attacker and neutraliser Qwen2.5-7B). $\bar V$: visibility share of the unmodified target page. Presentation share: fraction of the visibility gain of each rewrite group that disappears when the rewritten page is restated as facts (Sec.~\ref{sec:decomp}). $r$: split-half reliability of the per-page gap $M$. AUROC of $M$ against unmodified pages, measured ($S=2$ samples) and the ceiling for $S\to\infty$ predicted by the model of Sec.~\ref{sec:method-rel}. ASR: share of manipulative rewrites that raise target visibility by at least 20\%, without defence and with \sys{}-Replace ($M>\tau$); $p$: exact McNemar test. $^\dagger$AWQ-quantised, 120 test queries.}
\label{tab:engines}
\footnotesize
\setlength{\tabcolsep}{3.3pt}
\begin{tabular}{lrr ccc c cccc ccc}
\toprule
& & & \multicolumn{3}{c}{presentation share (\%)} & & \multicolumn{4}{c}{AUROC of $M$} & \multicolumn{3}{c}{ASR, manipulative} \\
\cmidrule(lr){4-6}\cmidrule(lr){8-11}\cmidrule(lr){12-14}
engine & \#q & $\bar V$ & \textcolor{legitc}{legit.} & \textcolor{geoc}{GEO} & \textcolor{manipc}{manip.} & $r$ & manip. & $S{\to}\infty$ & legit. & $S{\to}\infty$ & none & Replace & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("engines", s, best={12: "all"})


# ---------------------------------------------------------------------------- Table: detection (main engine)
def detection():
    E = load(RES / "eval" / f"{MAIN}.json")
    D, X = E["detection"], E["cross_attacker"]
    lofo = E["lofo"]["mean"]
    spec = [("M", "Gap $M$ (ours)", "--", "M", "M"),
            ("amort_all", r"\quad amortised", "gap", "amort_all", "amort_lofo"),
            ("M_para", r"\quad paraphrase gap", "--", None, None),
            ("cgeo", "C-GEO-Guard-lite", "attack", "cgeo", "cgeo_lofo"),
            ("sci", "SCI-lite (LLM)", "--", "sci", "sci"),
            ("guard", "Guard-style LLM", "--", "guard", "guard"),
            ("pi", "PI classifier", "PI", "pi", "pi"),
            ("ppl", "Perplexity", "--", "ppl", "ppl")]
    rows = []
    for k, name, sup, xk, lk in spec:
        if k not in D:
            continue
        d = D[k]
        cross = X.get(f"AUROC manipulative | {xk}") if xk else None
        lo = lofo.get(lk) if lk else None
        rows.append(f"{name} & {sup} & {f2(d['AUROC[manipulative vs benign]'])} & {f2(d['AUROC[held-out families]'])} & "
                    f"{f2(lo)} & {f2(cross)} & {f2(d['AUROC[legit vs benign]'])} " + r"\\")
        if k == "M_para":
            rows.append(r"\midrule")
    head = r"""\begin{table}[t]
\caption{Detection AUROC against unmodified pages (Qwen2.5-7B engine, 299 test queries). Held-out: the 9 GEO and manipulative families that no detector was trained on. LOFO: mean over leave-one-family-out retraining (Fig.~\ref{fig:lofo}). Other att.: manipulative pages written by Phi-3.5-mini, scored without retraining (150 queries). Legit.: legitimate edits, where a lower score is preferable. Sup.: training labels. The paraphrase gap uses a 150-query subset.}
\label{tab:detection}
\small
\setlength{\tabcolsep}{2.7pt}
\begin{tabular}{llccccc}
\toprule
detector & sup. & manip. & held-out & LOFO & other att. & legit.\,$\downarrow$ \\
\midrule
"""
    tail = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("detection", head + "\n".join(rows) + tail, best={2: "max", 3: "max", 4: "max", 5: "max", 6: "min"},
          ours=("(ours)", "amortised"))



# ---------------------------------------------------------------------------- Table: robustness of the decomposition
def robust():
    R = load(RES / "extra" / "robustness.json")
    A = load(RES / "extra" / "analysis.json")["Qwen/Qwen2.5-7B-Instruct"]
    D = R["decomposition"]
    mk = R["markers"]
    fams = [f for f in mk if not f.startswith("_")]
    kept_main = [mk[f]["main_N"] for f in fams]
    kept_alt = [mk[f]["alt_N"] for f in fams]
    rng = lambda xs: f"{100 * min(xs):.0f}--{100 * max(xs):.0f}"

    def cell(x):
        m, lo, hi = x["share_P"]
        return f"{100 * m:.0f} {{\\scriptsize[{100 * lo:.0f}, {100 * hi:.0f}]}}"

    def row(name, eng, nq, d, kept):
        return f"{name} & {eng} & {nq} & {cell(d['legit'])} & {cell(d['geo'])} & {cell(d['manipulative'])} & {kept} \\\\"

    main_all = {g: {"share_P": A["group"][g]["share_P"]} for g in ("legit", "geo", "manipulative")}
    rows = [
        row("main setting", "Q2.5-7B", A["n_queries"], main_all, rng(kept_main)),
        row("\\quad audited queries", "Q2.5-7B", 150, D["all_audited|Qwen2.5-7B"], rng(kept_main)),
        row("\\quad fact recall $\\geq 0.8$", "Q2.5-7B", 150, D["recall>=0.8|Qwen2.5-7B"], rng(kept_main)),
        "\\midrule",
        row("second attacker", "Q2.5-7B", D["alt_attacker|Qwen2.5-7B"]["n_queries"], D["alt_attacker|Qwen2.5-7B"], "--"),
        row("second attacker", "Q3-8B", D["alt_attacker|Qwen3-8B"]["n_queries"], D["alt_attacker|Qwen3-8B"], "--"),
        row("second neutraliser", "Q2.5-7B", D["alt_neutraliser|Qwen2.5-7B"]["n_queries"], D["alt_neutraliser|Qwen2.5-7B"], rng(kept_alt)),
    ]
    s = r"""\begin{table}[t]
\caption{Robustness of the decomposition: presentation-borne share (\%) of the visibility gain per group, with 95\% query-cluster CIs. Fact recall $\geq0.8$: only pairs whose original and rewritten pages both keep at least 80\% of their atomic facts after restatement. Second attacker: Phi-3.5-mini rewrites. Second neutraliser: Qwen3-8B. Rhetoric kept: share of manipulative pages whose restatement still contains a surface marker of the family (range over the five families).}
\label{tab:robust}
\footnotesize
\setlength{\tabcolsep}{2.0pt}
\begin{tabular}{llcccccc}
\toprule
setting & engine & \#q & \textcolor{legitc}{legit.} & \textcolor{geoc}{GEO} & \textcolor{manipc}{manip.} & rhet.\ kept \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("robust", s, ours=("main setting",))


# ---------------------------------------------------------------------------- Table: defences (main engine, 150 q)
def defences():
    E = load(RES / "eval" / f"{MAIN}.json")
    B = E["baselines_subset"]["defenses"]
    P = E["baselines_subset"]["asr_paired_vs_neutralens"]
    L = E["defense_cost"]["by_defense"]
    spec = [("none", "No defence", "--"),
            ("shield", "Shield prompt~\\cite{li2026geodefender}", "prompt"),
            ("ignore_instr", "Instruction hierarchy~\\cite{wallace2024hierarchy}", "prompt"),
            ("spotlight", "Spotlighting~\\cite{hines2024spotlighting}", "prompt"),
            ("paraphrase_all", "Paraphrase all~\\cite{jain2023baseline}", "rewrite all"),
            ("neutral_all", "\\sys{}-All", "rewrite all"),
            ("filter:ppl", "Perplexity", "remove"),
            ("filter:pi", "PI classifier", "remove"),
            ("filter:guard", "Guard-style LLM", "remove"),
            ("filter:sci", "SCI-lite", "remove"),
            ("filter:cgeo", "C-GEO-Guard-lite", "remove"),
            ("filter:M_fixed", "Gap $M>\\tau$", "remove"),
            ("replace:pi", "PI classifier", "replace"),
            ("replace:cgeo", "C-GEO-Guard-lite", "replace"),
            ("replace:amort_all", "Amortised gap", "replace"),
            ("replace_para:M_para", "Generic-paraphrase gap", "replace"),
            ("replace:M", "Gap, FPR-calibrated", "replace"),
            ("replace:M_fixed", "Gap $M>\\tau$ (\\sys{}-Replace)", "replace")]
    # Holm correction over all paired comparisons against NeutraLens-Replace
    ks = [k for k in P if k != "replace:M_fixed"]
    order = sorted(ks, key=lambda k: P[k]["p_mcnemar"])
    holm, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * P[k]["p_mcnemar"]))
        holm[k] = run
    rows, last = [], None
    for k, name, act in spec:
        if k not in B:
            continue
        v = B[k]
        if last is not None and act != last:
            rows.append("\\midrule")
        last = act
        ci = v.get("ASR_manipulative_ci") or [None, None]
        dq = v.get("benign_quality_delta_vs_none")
        dqs = "--" if not dq else f"{dq['delta']:+.2f}" if isinstance(dq, dict) and "delta" in dq else (
            f"{dq[0]:+.2f}" if isinstance(dq, (list, tuple)) else "--")
        lat = L.get(k, {}).get("latency online s")
        pv = pval(holm[k]) if k in holm else "--"
        rows.append(f"{name} & {act} & {f2(v['ASR_manipulative'])} [{f2(ci[0])}, {f2(ci[1])}] & "
                    f"{100 * v['gain_manipulative']:.1f} & {f2(v['ASR_geo'])} & "
                    f"{f2(v['ASR_fact'])} & {f2(v.get('legit_gain_retention'))} & {dqs} & {pct(v.get('benign_flag_rate'))} & "
                    f"{f2(lat, 1)} & {pv} \\\\")
    s = r"""\begin{table*}[t]
\caption{Defences on the main engine, all evaluated on the same 150 test queries. ASR at the 20\% threshold for manipulative, GEO and false-fact rewrites; the false-fact rewrite has zero mean gain (Fig.~\ref{fig:decomp}), so its ASR is the noise floor of the metric. Legit.\ kept: retained fraction of the visibility gain of legitimate edits. $\Delta Q$: change of LLM-judged answer quality (1--10) on clean queries. Flagged: share of unmodified target pages that the detector flags. Latency: batch-1 wall-clock seconds per query when nothing is precomputed (no defence: 3.5\,s; -- not timed). Gain: mean visibility gain of manipulative rewrites in percentage points of PAWC share, a threshold-free counterpart of ASR (removal sets the page's visibility to zero, so it can be negative). $p_{\text{Holm}}$: exact McNemar test of manipulative ASR against \sys{}-Replace, Holm-corrected over all 19 paired comparisons (the 17 rows above and two further amortised-detector variants).}
\label{tab:defences}
\footnotesize
\setlength{\tabcolsep}{2.7pt}
\begin{tabular}{llcccccccrr}
\toprule
defence & action & ASR manip. [95\% CI] & gain & GEO & false fact & legit.\ kept & $\Delta Q$ clean & flagged \% & latency (s) & $p_{\text{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("defences", s, best={2: "min", 4: "min"}, ours=("\\sys{}",))


# ---------------------------------------------------------------------------- Table: agent hijacking
def agent():
    A = load(RES / "extra" / "analysis.json")["agent"]
    names = {"none": "No defence", "rqa": "Root-query anchoring",
             "neutral_plan": "Planning gap (ours)", "neutral_all": "Fact-only reading"}
    rows = []
    for d, nm in names.items():
        cells = [nm]
        for full, sl in (("Qwen/Qwen2.5-7B-Instruct", "Qwen_Qwen2_5_7B_Instruct"), ("Qwen/Qwen3-8B", "Qwen_Qwen3_8B")):
            R = load(RES / "agent" / f"{sl}.json")
            a = A[full][f"attack|{d}"]
            r = R[f"attack|{d}"]
            ci = r["contamination_ci95"]
            cells += [pct(a["hijack"]), pct(a["brand_in_followups"]), pct(a['contamination'])]
        rows.append(" & ".join(cells) + " \\\\")
    s = r"""\begin{table}[t]
\caption{Research-agent hijacking (150 test queries, 2 planning rounds). Hijack: an attacker page is read or the attacker phrase or brand appears in a follow-up query; follow-up: the brand or phrase appears in the agent's own queries; contam.: the brand appears in the final report. All values in \%.}
\label{tab:agent}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lccc|ccc}
\toprule
& \multicolumn{3}{c|}{Qwen2.5-7B agent} & \multicolumn{3}{c}{Qwen3-8B agent} \\
defence & hijack & follow-up & contam. & hijack & follow-up & contam. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("agent", s, best={3: "min", 6: "min"}, ours=("(ours)", "Fact-only reading"))


# ---------------------------------------------------------------------------- appendix tables
def preservation():
    P = load(RES / "preservation.json")["by_family"]
    Pa = load(RES / "preservation_alt.json")
    groups = {"original page": ["benign"], "legitimate": ["clarity", "fluency", "structure"],
              "GEO": ["authoritative", "technical_terms", "unique_words", "keyword_stuffing", "statistics", "quotation",
                      "cite_sources"],
              "manipulative": ["hidden_instruction", "comparative_disparagement", "fake_endorsement", "citation_hook",
                               "emotional_urgency"], "false fact": ["fact_poison"]}

    def agg(src, fams, key):
        xs = [(src[f][key], src[f]["n"]) for f in fams if f in src and src[f].get(key) is not None]
        return sum(a * n for a, n in xs) / sum(n for _, n in xs) if xs else None
    alt_n = Pa.get("alt_neutralizer", {}).get("by_family", {}) if isinstance(Pa, dict) else {}
    alt_a = Pa.get("alt_attacker", {}).get("by_family", {}) if isinstance(Pa, dict) else {}
    rows = []
    for g, fams in groups.items():
        rows.append(f"{g} & {f2(agg(P, fams, 'fact_recall'))} & {f2(agg(P, fams, 'fact_precision'))} & "
                    f"{f2(agg(alt_n, fams, 'fact_recall'))} & {f2(agg(alt_n, fams, 'fact_precision'))} & "
                    f"{f2(agg(alt_a, fams, 'fact_recall'))} & {f2(agg(alt_a, fams, 'fact_precision'))} \\\\")
    s = r"""\begin{table}[!htbp]
\caption{Fact-preservation audit of the neutraliser (bidirectional NLI over atomic facts, 150 test queries). Recall: facts of the page entailed by $\mathcal{N}(d)$; precision: statements of $\mathcal{N}(d)$ entailed by the page. Columns: main neutraliser (Qwen2.5-7B), alternative neutraliser (Qwen3-8B), and the main neutraliser on pages of the second attacker (Phi-3.5-mini).}
\label{tab:preservation}
\small
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{main} & \multicolumn{2}{c}{alt.\ neutraliser} & \multicolumn{2}{c}{alt.\ attacker} \\
pages & rec. & prec. & rec. & prec. & rec. & prec. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("preservation", s)


def adaptive():
    A = load(RES / "adaptive.json")
    names = {"adapt_inject": "Injection aimed at $\\mathcal{N}$", "adapt_factify": "Fabricated facts",
             "adapt_evade_hidden_instruction": "Evasion (AI-directed)",
             "adapt_evade_fake_endorsement": "Evasion (endorsement)"}
    rows = []
    for k, nm in names.items():
        a = A[k]
        sv = a.get("injection_survival_rate")
        rows.append(f"{nm} & {f2(a['none']['ASR'])} & {f2(a['neutralens_exact']['ASR'])} & "
                    f"{f2(a['neutralens_amortised']['ASR'])} & {f2(a['neutral_all']['ASR'])} & "
                    f"{pct(a['amortised_flag_rate'])} & {'--' if sv is None else pct(sv)} \\\\")
    s = r"""\begin{table}[!htbp]
\caption{Adaptive attacks (main engine, 150 test queries). ASR under no defence, \sys{}-Replace with the exact gap, with the amortised detector, and \sys{}-All; amortised flag rate (\%); surv.\ (\%): neutral versions that still contain the injected sentence. Evasion: manipulative pages rewritten until the amortised detector stops flagging them.}
\label{tab:adaptive}
\footnotesize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lcccccc}
\toprule
attack & none & exact & amort. & All & flag & surv. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("adaptive", s)


def families():
    """per-family decomposition on all four engines (appendix)"""
    A = load(RES / "extra" / "analysis.json")
    fams = ["clarity", "fluency", "structure", "authoritative", "technical_terms", "unique_words", "keyword_stuffing",
            "statistics", "quotation", "cite_sources", "hidden_instruction", "comparative_disparagement",
            "fake_endorsement", "citation_hook", "emotional_urgency", "fact_poison"]
    nice = {f: f.replace("_", " ") for f in fams}
    nice.update({"hidden_instruction": "AI-directed instruction", "comparative_disparagement": "disparagement",
                 "fact_poison": "false fact", "emotional_urgency": "urgency"})
    rows = []
    for f in fams:
        cells = [nice[f]]
        for full, _, _ in ENGS:
            v = A[full]["family"][f]
            pv = f"{100 * v['P'][0]:.1f}"
            if f != 'fact_poison' and v['G'][0] > 0.01 and v['P'][0] / v['G'][0] >= 0.5:
                pv = r"\textcolor{presorange}{\textbf{" + pv + "}}"
            cells += [f"{100 * v['G'][0]:.1f}", pv]
        rows.append(" & ".join(cells) + " \\\\")
        if f in ("structure", "cite_sources", "emotional_urgency"):
            rows.append("\\midrule")
    s = r"""\begin{table*}[t]
\caption{Total visibility gain $\Delta_f$ and its presentation-borne part $P_f$ per family (percentage points of PAWC share; test queries). Orange: presentation carries at least half of the gain. Engines: Qwen2.5-7B, Qwen3-8B, Granite-3.1-8B, Qwen2.5-32B.}
\label{tab:families}
\small
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{2}{c}{Q2.5-7B} & \multicolumn{2}{c}{Q3-8B} & \multicolumn{2}{c}{Granite} & \multicolumn{2}{c}{Q2.5-32B} \\
family & $\Delta$ & $P$ & $\Delta$ & $P$ & $\Delta$ & $P$ & $\Delta$ & $P$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("families", s)


def tau():
    E = load(RES / "eval" / f"{MAIN}.json")
    rows = [f"{t['tau']:.2f} & {f2(t['ASR_manipulative'])} & {f2(t['ASR_geo'])} & {pct(t['benign_flag_rate'])} & "
            f"{f2(t['benign_quality'])} & {f2(t['legit_gain_retention'])} \\\\" for t in E["tau_curve"]]
    s = r"""\begin{table}[!htbp]
\caption{Threshold $\tau$ of \sys{}-Replace (main engine, 299 test queries).}
\label{tab:tau}
\small
\begin{tabular}{cccccc}
\toprule
$\tau$ & ASR manip. & ASR GEO & flagged (\%) & $Q$ clean & legit.\ kept \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tau", s, ours=("0.05 &",))


if __name__ == "__main__":
    engines()
    detection()
    defences()
    agent()
    robust()
    preservation()
    adaptive()
    families()
    tau()
    print("tables ->", TAB)
