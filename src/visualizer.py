import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Настройка шрифтов и общего стиля
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"  # Для корректного отображения кириллицы
plt.rcParams["text.color"] = "#231F20"        # Фирменный темный тон Mastercard для текста
plt.rcParams["axes.labelcolor"] = "#231F20"

# --- 1. ЦВЕТОВАЯ ПАЛИТРА MASTERCARD ---
MC_RED = "#EB001B"
MC_ORANGE = "#FF5F00"
MC_YELLOW = "#F79E1B"
MC_CHARCOAL = "#231F20"
MC_LIGHT_BG = "#FFF2CC"  # Мягкий желтоватый фон для плашек

# --- 2. ПОДГОТОВКА ДАННЫХ ---
TOTAL_CARDS = 80000

categories = [
    "LOW\n(Genuine Consumer\nscore < 0.30)",
    "MEDIUM (Grey Zone)\n(Needs Investigation\n0.30 - 0.70)",
    "HIGH\n(Hidden Business\nscore >= 0.70)",
]

# Данные из ваших логов
counts_before = [71634, 6148, 2218]  # До рефайнера
counts_after = [75801, 333, 3866]    # После честного пайплайна

pct_before = [count / TOTAL_CARDS * 100 for count in counts_before]
pct_after = [count / TOTAL_CARDS * 100 for count in counts_after]

# --- 3. ПОСТРОЕНИЕ ГРАФИКА ---
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.set_facecolor("white")

# Столбцы в фирменных цветах
rects1 = ax.bar(
    x - width / 2,
    pct_before,
    width,
    label="До серой зоны (Базовая модель)",
    color=MC_ORANGE,
    edgecolor=MC_CHARCOAL,
    linewidth=0.8,
    alpha=0.85
)
rects2 = ax.bar(
    x + width / 2,
    pct_after,
    width,
    label="После серой зоны (Composite Pipeline)",
    color=MC_RED,
    edgecolor=MC_CHARCOAL,
    linewidth=0.8,
    alpha=0.95
)

# --- 4. ОФОРМЛЕНИЕ ПОД СЛАЙДЫ ---
ax.set_ylabel("Процент от общего пула карт (%)", fontsize=12, fontweight="bold", color=MC_CHARCOAL)
ax.set_title(
    "Эффект очистки Серой Зоны (Grey Zone Refiner Utility)\n"
    f"Распределение 80,000 карт Consumer Pool по уровням риска",
    fontsize=14,
    fontweight="bold",
    pad=25,
    color=MC_CHARCOAL,
)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight="bold", color=MC_CHARCOAL)
ax.set_ylim(0, 105)

# Легенда с кастомным оформлением
ax.legend(fontsize=11, loc="upper right", frameon=True, facecolor="white", edgecolor=MC_CHARCOAL)

# Сетка в тон
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
ax.grid(axis='x', visible =False)

# Функция для добавления точных бизнес-метрик над столбцами
def autolabel(rects, absolute_counts):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        count = absolute_counts[idx]
        label_text = f"{height:.2f}%\n({count:,} шт)"

        ax.annotate(
            label_text,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=MC_CHARCOAL
        )

autolabel(rects1, counts_before)
autolabel(rects2, counts_after)

# Фирменная инфо-плашка (Business Impact)
info_text = (
    "Бизнес-эффект решения:\n"
    f"• Серая зона сократилась в 18 раз (с ~6.1к до {counts_after[1]} карт)\n"
    f"• Дополнительно выявлено +1,648 скрытых бизнесов (High Risk)\n"
    "• 94.7% пула успешно верифицированы как легитимные"
)
props = dict(boxstyle="round,pad=0.6", facecolor=MC_LIGHT_BG, edgecolor=MC_YELLOW, alpha=0.9)
ax.text(
    1.45,
    55,
    info_text,
    fontsize=11,
    bbox=props,
    verticalalignment="top",
    fontweight="bold",
    color=MC_CHARCOAL
)

# Убираем лишние границы графика для современного вида
sns.despine(left=True, bottom=True)

plt.tight_layout()

# Сохранение
project_root = os.getcwd() 
output_dir = os.path.join(project_root, "outputs", "figures")

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "09_grey_zone_mastercard_style.png")
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Презентационный график сохранен: {output_path}")