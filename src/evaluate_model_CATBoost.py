import os
import joblib
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

# 2. On charge le modèle sauvegardé (on crée une boîte vide puis on charge)import os
from catboost import CatBoostClassifier

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.normpath(os.path.join(BASE_DIR, "..", "modeles", "modele_catboost.pkl"))

print(f"Chargement depuis : {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle introuvable : {MODEL_PATH}")

model_cat = joblib.load(MODEL_PATH)
print("✅ Modèle chargé avec succès !")
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
plt.title("Matrice de Confusion : directement depuis le modèle chargé")
plt.show()