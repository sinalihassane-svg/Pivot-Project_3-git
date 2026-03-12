from xgboost import XGBClassifier
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

print("Modèle XGBoost entraîné avec succès.")