
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

# pour sauvegarder le modèle après l'entraînement 
model_cat.save_model("../modèles/modele_cancer_final.cbm")
print("Modèle sauvegardé avec succès dans le dossier !")
