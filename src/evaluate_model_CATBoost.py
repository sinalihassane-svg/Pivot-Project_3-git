import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import os

# ─────────────────────────────────────────────
# 1. Chargement des données et du modèle
# ─────────────────────────────────────────────
from data_processing import X_test_final, y_test

# 1. Configuration du chemin dynamique
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
# Vérifie bien si ton dossier s'appelle 'modeles' ou 'modèles'
MODELS_DIR = os.path.join(dossier_racine, "modeles") 

# 2. Chargement correct avec joblib
MODEL_PATH = os.path.join(MODELS_DIR, 'modele_catboost.pkl')


model_cat = joblib.load(MODEL_PATH)
print("✅ Modèle CatBoost chargé avec succès via joblib.")


# ─────────────────────────────────────────────
# 2. Prédictions
# ─────────────────────────────────────────────
y_pred       = model_cat.predict(X_test_final)
y_pred_proba = model_cat.predict_proba(X_test_final)[:, 1]

# ─────────────────────────────────────────────
# 3. Métriques
# ─────────────────────────────────────────────
report_dict = classification_report(
    y_test, y_pred,
    target_names=['Saine (0)', 'Risque (1)'],
    output_dict=True
)
report_str = classification_report(
    y_test, y_pred,
    target_names=['Saine (0)', 'Risque (1)']
)
print("\n" + "="*50)
print("       RAPPORT DE PERFORMANCE")
print("="*50)
print(report_str)

cm             = confusion_matrix(y_test, y_pred)
fpr, tpr, _    = roc_curve(y_test, y_pred_proba)
roc_auc        = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision  = average_precision_score(y_test, y_pred_proba)

accuracy  = report_dict['accuracy']
f1_risque = report_dict['Risque (1)']['f1-score']
prec_risk = report_dict['Risque (1)']['precision']
rec_risk  = report_dict['Risque (1)']['recall']

# ─────────────────────────────────────────────
# 4. Dashboard — une seule figure propre
# ─────────────────────────────────────────────
DARK_BG   = "#0F1117"
PANEL_BG  = "#1A1D27"
ACCENT    = "#00D4FF"
GREEN     = "#00FF9F"
ORANGE    = "#FF6B35"
TEXT      = "#E8EAF0"
SUBTEXT   = "#8B8FA8"
GRID_COL  = "#2A2D3E"

plt.rcParams.update({
    'font.family':      'monospace',
    'text.color':       TEXT,
    'axes.labelcolor':  TEXT,
    'xtick.color':      SUBTEXT,
    'ytick.color':      SUBTEXT,
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   GRID_COL,
    'grid.color':       GRID_COL,
})

fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
fig.suptitle(
    "CATBOOST CLASSIFIER — TABLEAU DE BORD D'ÉVALUATION",
    fontsize=18, fontweight='bold', color=ACCENT,
    y=0.97, fontfamily='monospace'
)

gs = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.45, wspace=0.35,
    top=0.93, bottom=0.06,
    left=0.05, right=0.97
)

# ── Ligne 1 : 4 KPI cards ──────────────────────────────────────────────────
kpis = [
    ("ACCURACY",    f"{accuracy:.2%}",    GREEN,  "Score global"),
    ("AUC-ROC",     f"{roc_auc:.4f}",     ACCENT, "Discrimination"),
    ("F1 — RISQUE", f"{f1_risque:.4f}",   ORANGE, "Classe Risque (1)"),
    ("AVG PREC.",   f"{avg_precision:.4f}", "#C77DFF", "Courbe PR"),
]

for i, (label, value, color, subtitle) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, 0.72, value,    transform=ax.transAxes, ha='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(0.5, 0.38, label,   transform=ax.transAxes, ha='center',
            fontsize=9,  fontweight='bold', color=TEXT)
    ax.text(0.5, 0.15, subtitle, transform=ax.transAxes, ha='center',
            fontsize=8,  color=SUBTEXT)

# ── Ligne 2-3 : Matrice de confusion ───────────────────────────────────────
ax_cm = fig.add_subplot(gs[1:, 0:2])
ax_cm.set_facecolor(PANEL_BG)

sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Saine (0)', 'Risque (1)'],
    yticklabels=['Saine (0)', 'Risque (1)'],
    ax=ax_cm,
    linewidths=2, linecolor=DARK_BG,
    annot_kws={"size": 22, "weight": "bold", "color": "white"},
    cbar_kws={"shrink": 0.8}
)
ax_cm.set_title("MATRICE DE CONFUSION", fontsize=12, fontweight='bold',
                color=ACCENT, pad=12)
ax_cm.set_xlabel("Prédiction",  fontsize=11, color=TEXT)
ax_cm.set_ylabel("Réel",        fontsize=11, color=TEXT)
ax_cm.tick_params(colors=TEXT, labelsize=10)

tn, fp, fn, tp = cm.ravel()
for (r, c), val in np.ndenumerate(cm):
    lbl = f"TP={val}" if (r==1 and c==1) else \
          f"FP={val}" if (r==0 and c==1) else \
          f"FN={val}" if (r==1 and c==0) else f"TN={val}"
    ax_cm.text(c + 0.5, r + 0.75, lbl,
               ha='center', va='center', fontsize=9,
               color=SUBTEXT, style='italic')

# ── Ligne 2 : Courbe ROC ───────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 2:])
ax_roc.set_facecolor(PANEL_BG)
ax_roc.grid(True, alpha=0.3)

ax_roc.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax_roc.plot(fpr, tpr, color=ACCENT, lw=2.5,
            label=f'AUC = {roc_auc:.4f}')
ax_roc.plot([0, 1], [0, 1], color=SUBTEXT, lw=1.5,
            linestyle='--', label='Aléatoire (AUC=0.5)')
ax_roc.set_xlim([0, 1])
ax_roc.set_ylim([0, 1.02])
ax_roc.set_xlabel("Taux de Faux Positifs", fontsize=10)
ax_roc.set_ylabel("Taux de Vrais Positifs", fontsize=10)
ax_roc.set_title("COURBE ROC", fontsize=12, fontweight='bold',
                 color=ACCENT, pad=12)
ax_roc.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT)

# ── Ligne 3 : Courbe Précision-Rappel ──────────────────────────────────────
ax_pr = fig.add_subplot(gs[2, 2:])
ax_pr.set_facecolor(PANEL_BG)
ax_pr.grid(True, alpha=0.3)

ax_pr.fill_between(recall, precision, alpha=0.15, color=GREEN)
ax_pr.plot(recall, precision, color=GREEN, lw=2.5,
           label=f'AP = {avg_precision:.4f}')
baseline = y_test.mean() if hasattr(y_test, 'mean') else np.mean(y_test)
ax_pr.axhline(y=baseline, color=SUBTEXT, lw=1.5, linestyle='--',
              label=f'Référence ({baseline:.2f})')
ax_pr.set_xlim([0, 1])
ax_pr.set_ylim([0, 1.05])
ax_pr.set_xlabel("Rappel (Recall)", fontsize=10)
ax_pr.set_ylabel("Précision",       fontsize=10)
ax_pr.set_title("COURBE PRÉCISION — RAPPEL", fontsize=12, fontweight='bold',
                color=GREEN, pad=12)
ax_pr.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
             labelcolor=TEXT)

# ── Signature ──────────────────────────────────────────────────────────────
import datetime
fig.text(0.97, 0.01,
         f"Généré le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
         ha='right', fontsize=7, color=SUBTEXT)

# ─────────────────────────────────────────────
# 5. Sauvegarde  ← AVANT plt.show()
# ─────────────────────────────────────────────
import os
os.makedirs("images", exist_ok=True)          # crée le dossier si absent
output_path = "images/evaluation_CATBoost_dashboard.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print(f"\n✅ Dashboard sauvegardé → {output_path}")

plt.show()