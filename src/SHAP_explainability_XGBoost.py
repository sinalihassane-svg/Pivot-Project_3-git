import os
import matplotlib.pyplot as plt
import shap
from train_model_XGBoost import model  # Import du modèle
from data_processing import X_test_final # Import des données

# --- ÉTAPE MANQUANTE : CALCUL DES VALEURS SHAP ---
print("Calcul des SHAP values (analyse en cours)...")
explainer = shap.TreeExplainer(model)
# C'est cette ligne qui crée la variable 'shap_values' qui te manque !
shap_values = explainer.shap_values(X_test_final) 

# --- CONFIGURATION DES CHEMINS (Ton code actuel) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
shap_output_dir = os.path.join(root_dir, "reports", "explanations")
os.makedirs(shap_output_dir, exist_ok=True)

# --- GÉNÉRATION ET SAUVEGARDE DU GRAPHIQUE BEESWARM ---
print("Génération du graphique Beeswarm...")
plt.figure(figsize=(12, 8))
# Maintenant 'shap_values' existe, donc ça ne plantera plus
shap.summary_plot(shap_values, X_test_final, show=False) 
plt.title("Analyse SHAP : Impact des facteurs sur le risque de cancer")

beeswarm_path = os.path.join(shap_output_dir, "xgboost_shap_beeswarm.png")
plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
plt.close() 

print(f"✅ Beeswarm enregistré dans : {beeswarm_path}")