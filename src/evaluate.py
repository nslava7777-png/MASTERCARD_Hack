from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

# ── ТЁМНАЯ ТЕМА MASTERCARD ────────────────────────────────────────────────────
BG        = "#0D0D0D"
FG        = "#FFFFFF"
MC_RED    = "#EB001B"
MC_ORANGE = "#FF5F00"
MC_YELLOW = "#F79E1B"
MC_GOLD   = "#FFD700"
GRID_COL  = "#2A2A2A"
PALETTE   = [MC_RED, MC_ORANGE, MC_YELLOW, MC_GOLD, "#FF8C00", "#DC143C"]

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


def calculate_metrics(y_true, y_pred, y_score):
    has_both = len(np.unique(y_true)) > 1
    return {
        "roc_auc":           float(roc_auc_score(y_true, y_score))           if has_both else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if has_both else None,
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def make_curve_payloads(y_true, proba_by_model):
    roc_data, pr_data = {}, {}
    for name, probs in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec, rec, _ = precision_recall_curve(y_true, probs)
        roc_data[name] = {"fpr": fpr, "tpr": tpr,
                          "auc": float(roc_auc_score(y_true, probs))}
        pr_data[name]  = {"precision": prec, "recall": rec,
                          "ap": float(average_precision_score(y_true, probs))}
    return roc_data, pr_data


def plot_confusion_matrix(y_true, y_pred, path, title):
    import matplotlib.colors as mcolors
    cm = confusion_matrix(y_true, y_pred)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mc", ["#1a0000", MC_RED, MC_ORANGE, MC_YELLOW])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["Consumer", "Business"],
                yticklabels=["Consumer", "Business"],
                linecolor=BG, linewidths=0.5,
                annot_kws={"color": FG, "fontweight": "bold"})
    ax.set_title(title, color=MC_YELLOW, fontweight="bold")
    ax.set_xlabel("Predicted", color=FG)
    ax.set_ylabel("Actual", color=FG)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_score_distribution(scores_dict, path, title, log_scale=True):
    colors = [MC_RED, MC_ORANGE, MC_YELLOW, MC_GOLD]
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, scores), color in zip(scores_dict.items(), colors):
        ax.hist(scores, bins=60, alpha=0.75, label=label,
                color=color, log=log_scale, density=True)
    ax.set_title(title, color=MC_YELLOW, fontweight="bold")
    ax.set_xlabel("Score")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc_curves(roc_data, path):
    colors = [MC_RED, MC_ORANGE, MC_YELLOW, MC_GOLD]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, d), color in zip(roc_data.items(), colors):
        ax.plot(d["fpr"], d["tpr"], color=color, lw=2,
                label=f"{name}  AUC={d['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#444444", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — OOF", color=MC_YELLOW, fontweight="bold")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_precision_recall_curves(pr_data, path):
    colors = [MC_RED, MC_ORANGE, MC_YELLOW, MC_GOLD]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, d), color in zip(pr_data.items(), colors):
        ax.plot(d["recall"], d["precision"], color=color, lw=2,
                label=f"{name}  AP={d['ap']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — OOF", color=MC_YELLOW, fontweight="bold")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_importance(fi, path, top_n=25):
    import matplotlib.colors as mcolors
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mc_bar", [MC_RED, MC_ORANGE, MC_YELLOW])
    colors = cmap(np.linspace(0, 1, len(fi)))
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.30)))
    ax.barh(fi["feature"], fi["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} Feature Importances",
                 color=MC_YELLOW, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_grey_zone_analysis(ensemble_scores, final_scores,
                             grey_low, grey_high, path):
    mask   = (ensemble_scores >= grey_low) & (ensemble_scores <= grey_high)
    before = ensemble_scores[mask]
    after  = final_scores[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grey Zone Refiner Analysis  [{grey_low}, {grey_high}]  n={mask.sum():,}",
        fontsize=13, fontweight="bold", color=MC_YELLOW,
    )

    axes[0].hist(before, bins=40, color=MC_ORANGE, alpha=0.85)
    axes[0].set_title("Before refiner (ensemble_score)", color=FG)
    axes[0].set_xlabel("Score")
    axes[0].axvline(0.5, color=MC_RED, ls="--", lw=1.5)

    axes[1].hist(after, bins=40, color=MC_RED, alpha=0.85)
    axes[1].set_title("After refiner (final_score)", color=FG)
    axes[1].set_xlabel("Score")
    axes[1].axvline(0.5, color=MC_YELLOW, ls="--", lw=1.5)

    delta = after - before
    axes[2].hist(delta, bins=40, color=MC_GOLD, alpha=0.85)
    axes[2].set_title("Score delta (after - before)", color=FG)
    axes[2].set_xlabel("Delta Score")
    axes[2].axvline(0, color=MC_RED, ls="--", lw=1.5)
    moved_up   = int((delta >  0.05).sum())
    moved_down = int((delta < -0.05).sum())
    axes[2].text(
        0.05, 0.92,
        f"UP   (+0.05): {moved_up:,} cards\nDOWN (-0.05): {moved_down:,} cards",
        transform=axes[2].transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="#1A0000",
                  edgecolor=MC_ORANGE, alpha=0.9),
        color=MC_YELLOW,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_consumer_score_breakdown(base_scores, dist_scores,
                                   iso_scores, final_scores, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Consumer Card Score Components",
                 fontsize=14, fontweight="bold", color=MC_YELLOW)
    panels = [
        (base_scores,  "Base model P(biz) [diagnostic]",      MC_ORANGE),
        (dist_scores,  "Distance score (centroid)",            MC_RED),
        (iso_scores,   "Isolation Forest anomaly",             MC_GOLD),
        (final_scores, "Final score (ensemble + grey refiner)", "#FF2244"),
    ]
    for ax, (scores, title, color) in zip(axes.ravel(), panels):
        ax.hist(scores, bins=60, color=color, alpha=0.85, log=True)
        ax.set_title(title, fontsize=10, color=FG)
        ax.set_xlabel("Score")
        p95 = float(np.percentile(scores, 95))
        p99 = float(np.percentile(scores, 99))
        ax.axvline(p95, color=MC_YELLOW, ls="--", lw=1.2, label=f"p95={p95:.3f}")
        ax.axvline(p99, color=FG,        ls=":",  lw=1.2, label=f"p99={p99:.3f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_top_suspicious_cards(card_ids, scores, path, top_n=50):
    df = (pd.DataFrame({"card": card_ids, "score": scores})
            .sort_values("score", ascending=False)
            .head(top_n))
    colors = [
        MC_RED    if s >= 0.70 else
        MC_ORANGE if s >= 0.50 else
        MC_YELLOW
        for s in df["score"]
    ]
    fig, ax = plt.subplots(figsize=(13, max(7, top_n * 0.24)))
    ax.barh(range(len(df)), df["score"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"...{str(c)[-8:]}" for c in df["card"]], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.50, color=MC_YELLOW, ls="--", lw=1.2, label="0.50")
    ax.axvline(0.70, color=MC_RED,    ls=":",  lw=1.5, label="0.70 HIGH")
    ax.set_xlabel("Final Score")
    ax.set_title(f"Top-{top_n} Suspected Hidden Businesses",
                 color=MC_YELLOW, fontweight="bold")
    ax.legend()
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_predictions(card_ids, scores, path, grey_low=0.30, high_thresh=0.70):
    def tier(s):
        if s >= high_thresh: return "HIGH"
        if s >= grey_low:    return "MEDIUM"
        return "LOW"
    df = pd.DataFrame({"card_number": card_ids, "score": scores})
    df["risk_tier"] = df["score"].apply(tier)
    df.sort_values("score", ascending=False).to_csv(path, index=False)