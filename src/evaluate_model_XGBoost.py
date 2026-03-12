# ==========================================
# Script d'évaluation amélioré (evaluate_model_XGBoost.py)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)

# --- 1. IMPORTATION DES DONNÉES ET DU MODÈLE (Obligatoire pour l'option A) ---
from data_processing import X_test_final, y_test
from train_model_XGBoost import model

print("Calcul des prédictions en cours...")

# --- 2. GÉNÉRATION DES PRÉDICTIONS ---
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)[:, 1] # Probabilités pour la classe positive

# ==========================================
#       AMÉLIORATION VISUELLE 1 : LES CHIFFRES
# ==========================================
print("\n" + "="*40)
print("### RAPPORT DE PERFORMANCE CLINIQUE ###")
print("="*40)

# Calcul des scores individuels
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Affichage formaté et espacé
print(f"{'Métrique':<25} | {'Valeur':<10}")
print("-" * 38)
print(f"{'Accuracy (Exactitude)':<25} | {accuracy:.2%}")
print(f"{'Précision':<25} | {precision:.2%}")
print(f"{'Recall (Sensibilité)':<25} | {recall:.2%}")
print(f"{'F1-Score':<25} | {f1:.2%}")
print(f"{'AUC-ROC':<25} | {auc:.4f}")
print("="*40)

# Affichage du rapport complet (texte)
print("\n### RAPPORT DÉTAILLÉ (Par classe) ###")
print(classification_report(y_test, y_pred, target_names=["Négatif (Sain)", "Positif (Cancer)"]))

# ==========================================
#       AMÉLIORATION VISUELLE 2 : LES GRAPHIQUES
# ==========================================
print("\nGénération des visualisations...")

# Préparation de la figure pour afficher les deux graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- GRAPHIQUE 1 : MATRICE DE CONFUSION VISUELLE ---
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Sain (0)", "Cancer (1)"], columns=["Prédit Sain (0)", "Prédit Cancer (1)"])

sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1, annot_kws={"size": 16})
ax1.set_title("Matrice de Confusion : Diagnostic Réel vs Prédit", fontsize=14, fontweight='bold')
ax1.set_xlabel("Diagnostic Prédit", fontsize=12)
ax1.set_ylabel("Diagnostic Réel", fontsize=12)

# --- GRAPHIQUE 2 : COURBE ROC (Receiver Operating Characteristic) ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Diagnostic Aléatoire (AUC=0.5)')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax2.set_ylabel("Taux de Vrais Positifs (Recall/Sensibilité)", fontsize=12)
ax2.set_title("Courbe ROC : Capacité de Séparation du Modèle", fontsize=14, fontweight='bold')
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(axis='both', alpha=0.3)

# Finalisation et affichage
plt.tight_layout()
print("\nAffichage des graphiques. Veuillez fermer la fenêtre pour terminer le script.")
plt.show()