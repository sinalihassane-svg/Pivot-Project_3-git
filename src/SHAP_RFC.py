#%%
import sys
import os
sys.path.append("src")

# ── Chemin vers le dossier images/ ───────────────────────────────────────────
DIR_SRC    = os.path.dirname(os.path.abspath(__file__))
DIR_ROOT   = os.path.dirname(DIR_SRC)
DIR_IMAGES = os.path.join(DIR_ROOT, "images")
os.makedirs(DIR_IMAGES, exist_ok=True)

def img(filename):
    """Retourne le chemin complet vers images/filename"""
    return os.path.join(DIR_IMAGES, filename)

import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from data_processing import X_test_final

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0D1117"
CARD   = "#161B22"
BORDER = "#21262D"
TEAL   = "#00C9A7"
BLUE   = "#1F6FEB"
RED    = "#F85149"
ORANGE = "#F0883E"
WHITE  = "#E6EDF3"
GREY   = "#8B949E"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : BORDER,
    "axes.labelcolor"  : WHITE,
    "xtick.color"      : GREY,
    "ytick.color"      : WHITE,
    "text.color"       : WHITE,
    "grid.color"       : BORDER,
    "grid.linewidth"   : 0.6,
    "font.family"      : "monospace",
})

# ── Chargement & SHAP ────────────────────────────────────────────────────────
# 1. On définit les chemins dynamiques
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
MODELS_DIR = os.path.join(dossier_racine, "modeles")

MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')


rf_model = joblib.load(MODEL_PATH)
print(f"✅ Modèle chargé pour SHAP depuis : {MODEL_PATH}")
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_final)

print(f"Type shap_values : {type(shap_values)}")
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

ev       = explainer.expected_value
base_val = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

feature_names = X_test_final.columns.tolist()
X_arr         = X_test_final.values

# ── Données calculées ────────────────────────────────────────────────────────
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
order         = np.argsort(mean_abs_shap)
TOP_N         = min(15, len(feature_names))
top_idx       = order[-TOP_N:]
top_names     = [feature_names[i] for i in top_idx]
top_vals      = mean_abs_shap[top_idx]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — IMPORTANCE GLOBALE
# ════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(13, 8), facecolor=BG)
fig1.suptitle("SHAP — IMPORTANCE GLOBALE DES VARIABLES",
              fontsize=13, fontweight="bold", color=TEAL, y=0.97)
fig1.text(0.5, 0.925, "Valeur SHAP absolue moyenne · Contribution au risque de cancer du col",
          ha="center", fontsize=9, color=GREY)

y_pos   = np.arange(TOP_N)
max_val = top_vals.max()

for i, (y, v) in enumerate(zip(y_pos, top_vals)):
    ax.barh(y, max_val * 1.05, color=BORDER, height=0.55, zorder=1)
    ratio = v / max_val
    color = (ratio * 0.0 + (1 - ratio) * 0.122,
             ratio * 0.788 + (1 - ratio) * 0.435,
             ratio * 0.655 + (1 - ratio) * 0.922)
    ax.barh(y, v, color=color, height=0.55, zorder=2, alpha=0.92)
    ax.text(v + max_val * 0.015, y, f"{v:.4f}",
            va="center", fontsize=8.5, color=WHITE, fontweight="bold")
    rank_color = TEAL if i >= TOP_N - 3 else (BLUE if i >= TOP_N - 7 else GREY)
    ax.barh(y, max_val * 0.008, left=-max_val * 0.06,
            color=rank_color, height=0.55, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_names, fontsize=9.5)
ax.set_xlim(-max_val * 0.07, max_val * 1.18)
ax.set_xlabel("Valeur SHAP absolue moyenne", fontsize=10, labelpad=10)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, alpha=0.25)

for label, color, x in [("Top 3", TEAL, 0.72), ("Top 7", BLUE, 0.80), ("Autres", GREY, 0.88)]:
    fig1.text(x, 0.02, f"▌ {label}", color=color, fontsize=8, ha="center")

plt.tight_layout(rect=[0, 0.04, 1, 0.92])
plt.savefig(img("shap_importance_bar.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"✅ Sauvegardé : {img('shap_importance_bar.png')}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — BEESWARM CUSTOM
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 9), facecolor=BG)
fig2.suptitle("SHAP — DISTRIBUTION DE L'IMPACT PAR VARIABLE",
              fontsize=13, fontweight="bold", color=TEAL, y=0.97)
fig2.text(0.5, 0.925,
          "Chaque point = un patient · Couleur = valeur de la feature (froid→chaud)",
          ha="center", fontsize=9, color=GREY)

cmap = plt.get_cmap("coolwarm")
np.random.seed(42)

for plot_i, feat_i in enumerate(top_idx):
    sv      = shap_vals[:, feat_i]
    fv      = X_arr[:, feat_i]
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    jitter  = np.random.uniform(-0.18, 0.18, size=len(sv))
    ax2.scatter(sv, plot_i + jitter, c=cmap(fv_norm), s=12,
                alpha=0.6, linewidths=0, zorder=2)
    ax2.axhline(plot_i, color=BORDER, lw=0.5, zorder=1)

ax2.set_yticks(range(TOP_N))
ax2.set_yticklabels(top_names, fontsize=9.5)
ax2.axvline(0, color=GREY, lw=1.2, linestyle="--", alpha=0.6)
ax2.set_xlabel("Valeur SHAP  (← réduit le risque  |  augmente le risque →)",
               fontsize=10, labelpad=10)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.tick_params(axis="y", length=0)
ax2.xaxis.grid(True, alpha=0.2)

sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax2, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Valeur normalisée de la feature", fontsize=8, color=GREY)
cbar.ax.yaxis.set_tick_params(color=GREY)
cbar.outline.set_edgecolor(BORDER)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GREY, fontsize=7)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Faible", "Moyen", "Élevé"])

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
plt.savefig(img("shap_beeswarm.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"✅ Sauvegardé : {img('shap_beeswarm.png')}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — WATERFALL CUSTOM
# ════════════════════════════════════════════════════════════════════════════
idx        = 0    # ← change pour analyser un autre patient
sv_patient = shap_vals[idx]
fv_patient = X_arr[idx]
n_show     = min(12, len(feature_names))
abs_ord    = np.argsort(np.abs(sv_patient))[-n_show:][::-1]
wf_names   = [feature_names[i] for i in abs_ord]
wf_shap    = sv_patient[abs_ord]
wf_fval    = fv_patient[abs_ord]
prediction = base_val + sv_patient.sum()

wf_shap_r  = wf_shap[::-1]
lefts      = base_val + np.concatenate([[0], np.cumsum(wf_shap_r[:-1])])

fig3, ax3 = plt.subplots(figsize=(14, 8), facecolor=BG)
fig3.suptitle(f"SHAP — EXPLICATION INDIVIDUELLE · PATIENT n°{idx}",
              fontsize=13, fontweight="bold", color=TEAL, y=0.97)
fig3.text(0.5, 0.925,
          f"Prédiction finale : {prediction:.4f}  ·  Base (moyenne) : {base_val:.4f}",
          ha="center", fontsize=9, color=GREY)

y_wf       = np.arange(n_show)
wf_names_r = wf_names[::-1]
wf_fval_r  = wf_fval[::-1]

for i, (y, sv, left, fname, fval) in enumerate(
        zip(y_wf, wf_shap_r, lefts, wf_names_r, wf_fval_r)):
    color = TEAL if sv > 0 else RED
    ax3.barh(y, sv, left=left, color=color, height=0.55, alpha=0.88, zorder=2)
    if i < n_show - 1:
        ax3.plot([left + sv, left + sv], [y + 0.275, y + 0.725],
                 color=BORDER, lw=1, zorder=1)
    ha   = "left" if sv > 0 else "right"
    xoff = left + sv + (0.003 if sv > 0 else -0.003)
    ax3.text(xoff, y, f"{sv:+.4f}", va="center", ha=ha,
             fontsize=8, color=color, fontweight="bold")
    ax3.text(-0.002, y, f"= {fval:.2f}", va="center", ha="right",
             fontsize=7.5, color=GREY, transform=ax3.get_yaxis_transform())

ax3.axvline(base_val,   color=GREY,   lw=1.2, linestyle=":",  alpha=0.7,
            label=f"Base : {base_val:.4f}")
ax3.axvline(prediction, color=ORANGE, lw=1.8, linestyle="--", alpha=0.9,
            label=f"Prédiction : {prediction:.4f}")
ax3.set_yticks(y_wf)
ax3.set_yticklabels(wf_names_r, fontsize=9.5)
ax3.set_xlabel("Valeur SHAP cumulée", fontsize=10, labelpad=10)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.tick_params(axis="y", length=0)
ax3.xaxis.grid(True, alpha=0.2)
ax3.legend(fontsize=9, loc="lower right",
           facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)

risk_label = "RISQUE ELEVE" if prediction > 0.5 else "FAIBLE RISQUE"
risk_color = RED if prediction > 0.5 else TEAL
fig3.text(0.97, 0.50, risk_label, ha="right", va="center",
          fontsize=10, fontweight="bold", color=risk_color, rotation=90)

plt.tight_layout(rect=[0, 0.02, 0.96, 0.92])
plt.savefig(img("shap_waterfall.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"✅ Sauvegardé : {img('shap_waterfall.png')}  (patient n°{idx})")
#%%