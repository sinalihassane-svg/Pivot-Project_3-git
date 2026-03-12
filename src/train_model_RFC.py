"""
Random Forest Classifier - Entraînement et Évaluation
Outil d'aide à la décision clinique - Risque de cancer du col de l'utérus
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing as dp
data_train = dp.X_train_final
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",   # compense le déséquilibre des classes
    random_state=42,
    n_jobs=-1
)
 
rf_model.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès.\n")
 
# ─────────────────────────────────────────────
# 4. PRÉDICTIONS
# ─────────────────────────────────────────────
y_pred       = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
 
# ─────────────────────────────────────────────
# 5. MÉTRIQUES DE PERFORMANCE
# ─────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)
 
print("=" * 45)
print("       RÉSULTATS - DONNÉES DE TEST")
print("=" * 45)
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 45)
 
print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred, target_names=["Non à risque", "À risque"]))
 
# Validation croisée (5-fold) sur le jeu d'entraînement
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC CV 5-fold : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
 
# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Évaluation du modèle Random Forest - Cancer du col de l'utérus", fontsize=14)
 
# --- 6a. Courbe ROC ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("Taux de faux positifs")
axes[0].set_ylabel("Taux de vrais positifs")
axes[0].set_title("Courbe ROC")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)
 
# --- 6b. Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
    xticklabels=["Non à risque", "À risque"],
    yticklabels=["Non à risque", "À risque"]
)
axes[1].set_xlabel("Prédit")
axes[1].set_ylabel("Réel")
axes[1].set_title("Matrice de confusion")
 
# --- 6c. Top 15 features importantes ---
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top15 = feat_importance.nlargest(15).sort_values()
axes[2].barh(top15.index, top15.values, color="#3498db")
axes[2].set_xlabel("Importance")
axes[2].set_title("Top 15 - Variables importantes")
axes[2].grid(alpha=0.3, axis="x")
 
plt.tight_layout()
plt.savefig("evaluation_random_forest.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Graphiques sauvegardés : evaluation_random_forest.png")
 
# ─────────────────────────────────────────────
# 7. SAUVEGARDE DU MODÈLE
# ─────────────────────────────────────────────
joblib.dump(rf_model, "random_forest_cervical_cancer.pkl")
print("💾 Modèle sauvegardé : random_forest_cervical_cancer.pkl")