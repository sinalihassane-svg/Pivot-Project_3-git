
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from data_processing import X_train_final, y_train_balanced
# Création du modèle avec les hyperparamètres validés
model_cat = CatBoostClassifier(
    iterations=500,          # Nombre d'arbres de décision
    learning_rate=0.05,      # Vitesse d'ajustement
    depth=6,                 # Complexité de chaque arbre
    verbose=100,             # Affiche un résumé tous les 100 cycles
    random_seed=42           # Pour avoir des résultats reproductibles
)

# Entraînement sur les données nettoyées par ton ami
print("Lancement de l'entraînement du modèle...")
model_cat.fit(X_train_final, y_train_balanced)
# ... (ton code actuel d'entraînement)
model_cat.fit(X_train_final, y_train_balanced)

import os
import joblib

# Création du chemin dynamique sécurisé
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
dossier_modeles = os.path.join(dossier_racine, "modeles") 
os.makedirs(dossier_modeles, exist_ok=True)

# Sauvegarde
chemin_sauvegarde = os.path.join(dossier_modeles, "modele_catboost.pkl")
joblib.dump(model_cat, chemin_sauvegarde)
print(f"✅ Modèle CatBoost Classifier sauvegardé sous : {chemin_sauvegarde}")
