
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import X_train_final, y_train_balanced

# Entraînement
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_final, y_train_balanced)

import os

# Création du chemin dynamique sécurisé
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
dossier_modeles = os.path.join(dossier_racine, "modeles") 
os.makedirs(dossier_modeles, exist_ok=True)

# Sauvegarde
chemin_sauvegarde = os.path.join(dossier_modeles, "random_forest_model.pkl")
joblib.dump(rf_model, chemin_sauvegarde)
print(f"✅ Modèle Random Forest Classifier sauvegardé sous : {chemin_sauvegarde}")
