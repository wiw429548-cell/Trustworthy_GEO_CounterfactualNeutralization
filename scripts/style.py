"""Shared figure style: ACM sans font (Linux Biolinum), the paper palette and small helpers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, DATA, FIG = ROOT / "results", ROOT / "data", ROOT / "assets" / "figures"
ICON = ROOT / "assets" / "icons"
FONT_DIR = Path("C:/Users/Administrator/AppData/Local/Programs/MiKTeX/fonts/opentype/public/libertine")

FAM = "Arial" if any(x.name == "Arial" for x in fm.fontManager.ttflist) else "DejaVu Sans"
for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf"):
    if (FONT_DIR / f).exists():
        fm.fontManager.addfont(str(FONT_DIR / f))
        FAM = "Linux Biolinum O"
plt.rcParams.update({
    "font.family": FAM, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#1f2328", "text.color": "#1f2328",
    "legend.frameon": False, "savefig.dpi": 300})
if FAM not in ("Arial", "DejaVu Sans"):
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": FAM, "mathtext.it": FAM + ":italic",
                         "mathtext.bf": FAM + ":bold", "mathtext.cal": "cmsy10"})

# paper palette (same hex values as the \definecolor block in main.tex)
OURS = "#2a78d6"      # fact-borne / NeutraLens
PRES = "#eb6834"      # presentation-borne
LEGIT = "#1baf7a"
GEO = "#eda100"
MANIP = "#e34948"
VIOLET = "#4a3aa7"
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8a8984", "#e4e3df"
PANEL = "#f6f6f4"     # light panel background
OURS_BG, PRES_BG, MANIP_BG = "#e8f0fb", "#fde7dd", "#fbe3e3"
GROUP_COL = {"legit": LEGIT, "geo": GEO, "manipulative": MANIP, "fact": MUTED}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=220)
    plt.close(fig)
