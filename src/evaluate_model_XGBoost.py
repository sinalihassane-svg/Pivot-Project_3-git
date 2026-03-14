import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve)

# --- 1. IMPORTATION DES COMPOSANTS ---
# On récupère les données de test et le modèle entraîné depuis les autres fichiers
from data_processing import X_test_final, y_test
from train_model_XGBoost import model

# --- 2. CALCUL DES PRÉDICTIONS ---
# 'y_pred' pour les classes (0 ou 1) et 'y_pred_proba' pour le score de probabilité (nécessaire pour l'AUC)
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)[:, 1]

# --- 3. CALCUL DES MÉTRIQUES CLINIQUES ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred) # Très critique pour le diagnostic médical
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# --- 4. CRÉATION DU VISUEL COMPLET ---
# On crée une figure avec 3 sous-graphiques (1 ligne, 3 colonnes)
fig, (ax_table, ax_cm, ax_roc) = plt.subplots(1, 3, figsize=(20, 6))

# --- A. PANNEAU DE GAUCHE : TABLEAU DES MÉTRIQUES ---
ax_table.axis('off') # On cache les axes car on veut juste afficher du texte
text_metrics = (
    f"--- BILAN DES PERFORMANCES ---\n\n"
    f"Accuracy  : {accuracy:.2%}\n\n"
    f"Précision : {precision:.2%}\n\n"
    f"Recall    : {recall:.2%}\n\n"
    f"F1-Score  : {f1:.2%}\n\n"
    f"ROC-AUC   : {auc:.4f}"
)
# On place le texte au centre du premier panneau
ax_table.text(0.5, 0.5, text_metrics, fontsize=16, va='center', ha='center', 
              bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=1'))
ax_table.set_title("Scores du Modèle", fontsize=14, fontweight='bold')

# --- B. PANNEAU CENTRAL : MATRICE DE CONFUSION ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm, annot_kws={"size": 18})
ax_cm.set_title("Matrice de Confusion", fontsize=14, fontweight='bold')
ax_cm.set_xticklabels(['Prédit Sain', 'Prédit Cancer'])
ax_cm.set_yticklabels(['Réel Sain', 'Réel Cancer'])
ax_cm.set_xlabel("Prédiction du Modèle")
ax_cm.set_ylabel("Réalité Médicale")

# --- C. PANNEAU DE DROITE : COURBE ROC ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Ligne de base (aléatoire)
ax_roc.set_title("Courbe ROC", fontsize=14, fontweight='bold')
ax_roc.set_xlabel("Taux de Faux Positifs")
ax_roc.set_ylabel("Taux de Vrais Positifs (Recall)")
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.2)

# Ajustement de l'espacement pour éviter les chevauchements
plt.tight_layout()
# --- CRÉATION DU DOSSIER DE SORTIE (Chemin Absolu) ---
# On récupère le chemin du dossier où se trouve ce script (src)
script_dir = os.path.dirname(os.path.abspath(__file__))

# On définit le dossier reports à la racine (un niveau au dessus de src)
root_dir = os.path.dirname(script_dir)
output_dir = os.path.join(root_dir, "reports", "metrics")

# Création du dossier s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# --- ENREGISTREMENT ---
file_path = os.path.join(output_dir, "evaluation_summary.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"✅ Image enregistrée avec succès dans : {file_path}")
plt.show()
