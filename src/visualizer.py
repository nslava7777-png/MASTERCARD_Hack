import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── ТЁМНАЯ ТЕМА MASTERCARD ────────────────────────────────────────────────────
BG        = "#0D0D0D"
FG        = "#FFFFFF"
MC_RED    = "#EB001B"
MC_ORANGE = "#FF5F00"
MC_YELLOW = "#F79E1B"
MC_GOLD   = "#FFD700"
GRID_COL  = "#2A2A2A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    FG,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "text.color":        FG,
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  MC_ORANGE,
    "legend.labelcolor": FG,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
})
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_CARDS = 80000
categories = [
    "LOW\n(Genuine Consumer\nscore < 0.30)",
    "MEDIUM (Grey Zone)\n(Needs Investigation\n0.30 - 0.70)",
    "HIGH\n(Hidden Business\nscore >= 0.70)",
]
counts_before = [71634, 6148, 2218]
counts_after  = [75801, 333,  3866]
pct_before = [c / TOTAL_CARDS * 100 for c in counts_before]
pct_after  = [c / TOTAL_CARDS * 100 for c in counts_after]

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width / 2, pct_before, width,
                label="До серой зоны (Базовая модель)",
                color=MC_ORANGE, edgecolor=MC_YELLOW, linewidth=0.8, alpha=0.9)
rects2 = ax.bar(x + width / 2, pct_after, width,
                label="После серой зоны (Composite Pipeline)",
                color=MC_RED, edgecolor=MC_GOLD, linewidth=0.8, alpha=0.95)

ax.set_ylabel("Процент от общего пула карт (%)", fontsize=12,
              fontweight="bold")
ax.set_title(
    "Эффект очистки Серой Зоны (Grey Zone Refiner Utility)\n"
    f"Распределение {TOTAL_CARDS:,} карт Consumer Pool по уровням риска",
    fontsize=14, fontweight="bold", color=MC_YELLOW
)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y")
ax.grid(axis="x", visible=False)


def autolabel(rects, counts, col):
    for idx, rect in enumerate(rects):
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}%\n({counts[idx]:,} шт)",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=col
        )


autolabel(rects1, counts_before, MC_YELLOW)
autolabel(rects2, counts_after,  MC_GOLD)

info_text = (
    "Бизнес-эффект решения:\n"
    f"• Серая зона сократилась в 18 раз (с ~6.1к до {counts_after[1]} карт)\n"
    f"• Дополнительно выявлено +1,648 скрытых бизнесов\n"
    "• 94.7% пула верифицированы как легитимные"
)
props = dict(boxstyle="round,pad=0.6", facecolor="#1A0000",
             edgecolor=MC_ORANGE, alpha=0.95)
ax.text(1.45, 55, info_text, fontsize=11, bbox=props,
        verticalalignment="top", fontweight="bold", color=MC_YELLOW)

sns.despine(left=True, bottom=True)
plt.tight_layout()

output_dir = os.path.join(os.getcwd(), "outputs", "figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "09_grey_zone_mastercard_style.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"График сохранён: {output_path}")