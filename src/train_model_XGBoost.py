from xgboost import XGBClassifier
import joblib
import os
from data_processing import X_train_final, y_train_balanced
# Initialisation du modèle XGBoost
# On définit eval_metric pour éviter les avertissements de dépréciation
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Entraînement sur les données prétraitées et équilibrées
model.fit(X_train_final, y_train_balanced)


# --- Après model.fit(...) ---

# 1. Définir le chemin de sauvegarde (à la racine du projet)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
model_path = os.path.join(root_dir, "xgboost_cancer_model.pkl")

# 2. Enregistrer le modèle
joblib.dump(model, model_path)

print(f"✅ Modèle XGBoost sauvegardé avec succès sous : {model_path}")

print("Modèle XGBoost entraîné avec succès.")