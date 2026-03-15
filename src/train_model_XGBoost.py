from xgboost import XGBClassifier
import joblib
import os
from data_processing import X_train_final, y_train_balanced
import joblib
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



# Création du chemin dynamique sécurisé
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
dossier_modeles = os.path.join(dossier_racine, "modeles") 
os.makedirs(dossier_modeles, exist_ok=True)

# Sauvegarde
chemin_sauvegarde = os.path.join(dossier_modeles, "XGBoost_model.pkl")
joblib.dump(model, chemin_sauvegarde)
print(f"✅ Modèle XGBoost Classifier sauvegardé sous : {chemin_sauvegarde}")
