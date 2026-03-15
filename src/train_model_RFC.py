
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import X_train_final, y_train_balanced

# Entraînement
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_final, y_train_balanced)

# Sauvegarde du modèle
joblib.dump(rf_model, "../modèles/random_forest_model.pkl")
print("Modèle entraîné et sauvegardé sous 'random_forest_model.pkl'")
