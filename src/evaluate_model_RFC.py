#%%
import sys
sys.path.append("src")

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from data_processing import X_test_final, y_test

# ── Palette & style ──────────────────────────────────────────────────────────
BG        = "#0D1117"
CARD      = "#161B22"
BORDER    = "#21262D"
TEAL      = "#00C9A7"
BLUE      = "#1F6FEB"
RED       = "#F85149"
YELLOW    = "#D29922"
WHITE     = "#E6EDF3"
GREY      = "#8B949E"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : CARD,
    "axes.edgecolor"    : BORDER,
    "axes.labelcolor"   : WHITE,
    "xtick.color"       : GREY,
    "ytick.color"       : GREY,
    "text.color"        : WHITE,
    "grid.color"        : BORDER,
    "grid.linewidth"    : 0.6,
    "font.family"       : "monospace",
})

# ── Chargement & prédictions ─────────────────────────────────────────────────
rf_model     = joblib.load("random_forest_model.pkl")
y_pred       = rf_model.predict(X_test_final)
y_pred_proba = rf_model.predict_proba(X_test_final)[:, 1]

metrics = {
    "Accuracy" : accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall"   : recall_score(y_test, y_pred, zero_division=0),
    "F1-Score" : f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC"  : roc_auc_score(y_test, y_pred_proba),
}

# ── Layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle("RANDOM FOREST — RAPPORT D'ÉVALUATION CLINIQUE",
             fontsize=14, fontweight="bold", color=TEAL,
             y=0.97)
fig.text(0.5, 0.935, "Cancer du col de l'utérus · Détection de risque · Classification binaire",
         ha="center", fontsize=9, color=GREY)

gs = gridspec.GridSpec(2, 4, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.04, right=0.97,
                       top=0.90, bottom=0.06)

# ── 1. Cartes métriques ───────────────────────────────────────────────────────
metric_colors = [TEAL, BLUE, RED, YELLOW, TEAL]
metric_labels = list(metrics.keys())
metric_values = list(metrics.values())

for i, (name, val, color) in enumerate(zip(metric_labels, metric_values, metric_colors)):
    col = i if i < 4 else 3
    row = 0 if i < 4 else 0
    ax  = fig.add_subplot(gs[0, i % 4])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Fond de carte
    rect = FancyBboxPatch((0.05, 0.08), 0.90, 0.84,
                          boxstyle="round,pad=0.02",
                          linewidth=1.5, edgecolor=color,
                          facecolor=CARD, zorder=1)
    ax.add_patch(rect)

    # Barre de progression
    bar_bg = FancyBboxPatch((0.12, 0.15), 0.76, 0.10,
                            boxstyle="round,pad=0.01",
                            facecolor=BORDER, edgecolor="none", zorder=2)
    bar_fg = FancyBboxPatch((0.12, 0.15), 0.76 * val, 0.10,
                            boxstyle="round,pad=0.01",
                            facecolor=color, edgecolor="none",
                            alpha=0.85, zorder=3)
    ax.add_patch(bar_bg)
    ax.add_patch(bar_fg)

    # Textes
    ax.text(0.50, 0.72, f"{val:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color, zorder=4)
    ax.text(0.50, 0.50, name.upper(), ha="center", va="center",
            fontsize=8.5, color=GREY, zorder=4)

# ── 2. Matrice de confusion ───────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 0:2])
cm    = confusion_matrix(y_test, y_pred)
im    = ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0)

for i in range(2):
    for j in range(2):
        color = WHITE if cm[i, j] > cm.max() / 2 else GREY
        ax_cm.text(j, i, str(cm[i, j]),
                   ha="center", va="center",
                   fontsize=20, fontweight="bold", color=color)

labels = ["Négatif\n(Sain)", "Positif\n(Cancer)"]
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(labels, fontsize=9)
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(labels, fontsize=9, rotation=90, va="center")
ax_cm.set_xlabel("Prédiction", fontsize=10, labelpad=8)
ax_cm.set_ylabel("Réalité",    fontsize=10, labelpad=8)
ax_cm.set_title("MATRICE DE CONFUSION", fontsize=10, color=TEAL,
                fontweight="bold", pad=12, loc="left")

tn, fp, fn, tp = cm.ravel()
for (x, y_pos, label, c) in [(0, 0, "VN", TEAL), (1, 0, "FP", RED),
                               (0, 1, "FN", YELLOW), (1, 1, "VP", BLUE)]:
    ax_cm.text(x + 0.42, y_pos - 0.40, label,
               ha="center", va="center", fontsize=7,
               color=c, fontweight="bold",
               bbox=dict(facecolor=BG, edgecolor=c, boxstyle="round,pad=0.2", lw=0.8))

# ── 3. Courbe ROC ─────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 2:4])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_val      = metrics["ROC-AUC"]

ax_roc.fill_between(fpr, tpr, alpha=0.15, color=TEAL)
ax_roc.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"AUC = {auc_val:.4f}")
ax_roc.plot([0, 1], [0, 1], color=BORDER, lw=1.2, linestyle="--", label="Aléatoire (0.50)")

ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.01)
ax_roc.set_xlabel("Taux de Faux Positifs (1 - Spécificité)", fontsize=9, labelpad=8)
ax_roc.set_ylabel("Taux de Vrais Positifs (Sensibilité)",    fontsize=9, labelpad=8)
ax_roc.set_title("COURBE ROC", fontsize=10, color=TEAL,
                 fontweight="bold", pad=12, loc="left")
ax_roc.grid(True, alpha=0.3)

legend = ax_roc.legend(fontsize=9, loc="lower right",
                        facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)

# Annotation AUC
ax_roc.annotate(f"AUC\n{auc_val:.4f}",
                xy=(0.6, 0.4), xytext=(0.35, 0.25),
                fontsize=11, fontweight="bold", color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
                bbox=dict(facecolor=CARD, edgecolor=TEAL, boxstyle="round,pad=0.4"))

# ── Ligne de séparation ───────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.03, 0.97], [0.515, 0.515],
                           transform=fig.transFigure,
                           color=BORDER, lw=1))

plt.savefig("evaluation_RFC_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Dashboard sauvegardé : evaluation_RFC_dashboard.png")
#%%