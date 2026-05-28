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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Consumer", "Business"],
                yticklabels=["Consumer", "Business"])
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_score_distribution(scores_dict, path, title, log_scale=True):
    plt.figure(figsize=(10, 5))
    colors = ["steelblue", "darkorange", "green", "red"]
    for (label, scores), color in zip(scores_dict.items(), colors):
        plt.hist(scores, bins=60, alpha=0.65, label=label,
                 color=color, log=log_scale, density=True)
    plt.title(title); plt.xlabel("Score"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_roc_curves(roc_data, path):
    plt.figure(figsize=(7, 6))
    for name, d in roc_data.items():
        plt.plot(d["fpr"], d["tpr"], label=f"{name}  AUC={d['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves — OOF"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_precision_recall_curves(pr_data, path):
    plt.figure(figsize=(7, 6))
    for name, d in pr_data.items():
        plt.plot(d["recall"], d["precision"],
                 label=f"{name}  AP={d['ap']:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — OOF"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_feature_importance(fi, path, top_n=25):
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    colors = plt.cm.viridis_r(np.linspace(0.2, 0.85, len(fi)))
    plt.figure(figsize=(10, max(5, top_n * 0.30)))
    plt.barh(fi["feature"], fi["importance"], color=colors)
    plt.gca().invert_yaxis()
    plt.title(f"Top-{top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_grey_zone_analysis(ensemble_scores, final_scores,
                             grey_low, grey_high, path):
    mask   = (ensemble_scores >= grey_low) & (ensemble_scores <= grey_high)
    before = ensemble_scores[mask]
    after  = final_scores[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grey Zone Refiner Analysis  [{grey_low}, {grey_high}]  n={mask.sum():,}",
        fontsize=13, fontweight="bold",
    )

    axes[0].hist(before, bins=40, color="steelblue", alpha=0.8)
    axes[0].set_title("Before refiner (ensemble_score)")
    axes[0].set_xlabel("Score")
    axes[0].axvline(0.5, color="red", ls="--", lw=1.5)

    axes[1].hist(after, bins=40, color="darkorange", alpha=0.8)
    axes[1].set_title("After refiner (final_score)")
    axes[1].set_xlabel("Score")
    axes[1].axvline(0.5, color="red", ls="--", lw=1.5)

    delta = after - before
    axes[2].hist(delta, bins=40, color="green", alpha=0.8)
    axes[2].set_title("Score delta (after - before)")
    axes[2].set_xlabel("Delta Score")
    axes[2].axvline(0, color="red", ls="--", lw=1.5)
    moved_up   = int((delta >  0.05).sum())
    moved_down = int((delta < -0.05).sum())
    axes[2].text(
        0.05, 0.92,
        f"UP   (+0.05): {moved_up:,} cards\nDOWN (-0.05): {moved_down:,} cards",
        transform=axes[2].transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_consumer_score_breakdown(base_scores, dist_scores,
                                   iso_scores, final_scores, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Consumer Card Score Components",
                 fontsize=14, fontweight="bold")
    panels = [
        (base_scores,  "Base model P(biz) [diagnostic]", "steelblue"),
        (dist_scores,  "Distance score (centroid)",       "darkorange"),
        (iso_scores,   "Isolation Forest anomaly",        "green"),
        (final_scores, "Final score (ensemble + grey refiner)", "crimson"),
    ]
    for ax, (scores, title, color) in zip(axes.ravel(), panels):
        ax.hist(scores, bins=60, color=color, alpha=0.8, log=True)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Score")
        p95 = float(np.percentile(scores, 95))
        p99 = float(np.percentile(scores, 99))
        ax.axvline(p95, color="black", ls="--", lw=1.2, label=f"p95={p95:.3f}")
        ax.axvline(p99, color="gray",  ls=":",  lw=1.2, label=f"p99={p99:.3f}")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_top_suspicious_cards(card_ids, scores, path, top_n=50):
    df = (pd.DataFrame({"card": card_ids, "score": scores})
            .sort_values("score", ascending=False)
            .head(top_n))
    colors = [
        "crimson"     if s >= 0.70 else
        "darkorange"  if s >= 0.50 else
        "steelblue"
        for s in df["score"]
    ]
    plt.figure(figsize=(13, max(7, top_n * 0.24)))
    plt.barh(range(len(df)), df["score"], color=colors)
    plt.yticks(range(len(df)),
               [f"...{str(c)[-8:]}" for c in df["card"]], fontsize=7)
    plt.gca().invert_yaxis()
    plt.axvline(0.50, color="black", ls="--", lw=1.2, label="0.50")
    plt.axvline(0.70, color="red",   ls=":",  lw=1.2, label="0.70 HIGH")
    plt.xlabel("Final Score")
    plt.title(f"Top-{top_n} Suspected Hidden Businesses")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def save_predictions(card_ids, scores, path, grey_low=0.30, high_thresh=0.70):
    def tier(s):
        if s >= high_thresh: return "HIGH"
        if s >= grey_low:    return "MEDIUM"
        return "LOW"
    df = pd.DataFrame({"card_number": card_ids, "score": scores})
    df["risk_tier"] = df["score"].apply(tier)
    df.sort_values("score", ascending=False).to_csv(path, index=False)
