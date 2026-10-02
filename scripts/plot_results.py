"""Render docs/recall_at_5.png from results/bench.json."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = [("bm25", "BM25", "#2a78d6"), ("dense", "Dense (MiniLM)", "#eb6834"), ("hybrid", "Hybrid (RRF)", "#1baf7a")]
INDEXES = [("lines", False, "60-line windows"), ("lines", True, "60-line windows\n+ path header"),
           ("ast", False, "AST chunks"), ("ast", True, "AST chunks\n+ path/symbol header")]
PANELS = [("semantic", "Natural-language questions"), ("keyword", "Questions naming an identifier")]


def main(src="results/bench.json", out="docs/recall_at_5.png", metric="recall@5"):
    rows = json.loads(Path(src).read_text())
    get = {(r["chunker"], r["header"], r["mode"]): r["summary"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    width = 0.27
    for ax, (cat, title) in zip(axes, PANELS):
        for j, (mode, label, color) in enumerate(MODES):
            xs = [i + (j - 1) * width for i in range(len(INDEXES))]
            ys = [get[(c, h, mode)][cat][metric] for c, h, _ in INDEXES]
            bars = ax.bar(xs, ys, width - 0.03, color=color, label=label, zorder=2)
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.015, f"{y:.0%}", ha="center", fontsize=7, color="#444")
        n = get[(INDEXES[0][0], INDEXES[0][1], "bm25")][cat]["n"]
        ax.set_title(f"{title} (n={n})", fontsize=11, color="#222")
        ax.set_xticks(range(len(INDEXES)), [lab for *_, lab in INDEXES], fontsize=9, color="#444")
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color="#e6e6e6", zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", colors="#666", length=0)
    axes[0].set_ylabel(f"{metric.capitalize()} (right code in top 5)", color="#444")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
