import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
import os

# 1. On récupère les données (indispensable pour que SHAP puisse comparer les patientes)
from data_processing import X_test_final

# 2. On charge le modèle que tu as déjà entraîné
model_cat = CatBoostClassifier()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_cat.load_model(os.path.join(BASE_DIR, '..', 'modèles', 'modele_cancer_final.cbm'))
print("Modèle chargé pour l'analyse SHAP.")

# 3. Création de l'expliqueur SHAP
# On utilise TreeExplainer car CatBoost est un modèle d'arbres
explainer = shap.TreeExplainer(model_cat)

# 4. Calcul des SHAP values (cela peut prendre quelques secondes)
shap_values = explainer.shap_values(X_test_final)

# 5. Graphique 1 : Importance globale (Barres)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_final, plot_type="bar", show=False)
plt.title("Importance globale des facteurs de risque")
plt.show()

# 6. Graphique 2 : Impact détaillé (Beeswarm)
# C'est celui qui montre si une valeur haute augmente ou diminue le risque
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_final, show=False)
plt.show()