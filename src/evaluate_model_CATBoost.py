import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from data_processing import X_test_final, y_test

# 1. On réimporte les données nettoyées depuis le fichier de ton ami
from data_processing import X_test_final, y_test

# 2. On charge le modèle sauvegardé (on crée une boîte vide puis on charge)
model_cat = CatBoostClassifier()
model_cat.load_model("modele_cancer_final.cbm")
print("Modèle chargé et prêt pour l'évaluation.")

# 3. Évaluation
y_pred = model_cat.predict(X_test_final)

print("\n" + "="*40)
print("   RAPPORT DE PERFORMANCE")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Saine (0)', 'Risque (1)']))

# 4. Affichage de la Matrice de Confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Saine (0)', 'Risque (1)'])
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion (Chargée depuis le fichier)")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Génération des prédictions sur le jeu de test réel
y_pred = model_cat.predict(X_test_final)
y_pred_proba = model_cat.predict_proba(X_test_final)[:, 1] # Probabilités pour la courbe ROC

# 2. Affichage du Rapport de Classification
# Ce rapport donne la Précision, le Rappel (Recall) et le F1-Score
print("\n" + "="*40)
print("   RAPPORT DE PERFORMANCE (JEU DE TEST)")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Saine (0)', 'Risque (1)']))

# 3. Visualisation de la Matrice de Confusion
# Elle montre les Vrais Positifs, Faux Positifs, Vrais Négatifs et Faux Négatifs
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Saine (0)', 'Risque (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Matrice de Confusion : Diagnostic Final")
plt.show()

# 4. Courbe ROC et Score AUC
# L'AUC (Area Under Curve) mesure la performance globale du modèle (plus proche de 1, mieux c'est)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs (Recall)")
plt.title("Courbe ROC - Capacité de discrimination du modèle")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()