"""
=============================================================================
TESTS AUTOMATISÉS — Système de Soutien Clinique (XGBoost)
=============================================================================
Ce script est conçu pour être portable : il détecte automatiquement la racine
du projet peu importe l'endroit où il est exécuté.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import joblib

# =============================================================================
# GESTION ROBUSTE DES CHEMINS (PORTABILITÉ TOTALE)
# =============================================================================
# On définit la racine du projet comme étant le parent du dossier 'tests'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'modeles')

# Ajout dynamique de 'src' au PYTHONPATH
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import des données depuis ton module de traitement
try:
    from data_processing import (
        X_test_final, y_test, 
        X_train_final, y_train_balanced,
        scaler, imputer
    )
except ImportError as e:
    print(f"❌ Erreur : Impossible de charger 'data_processing' depuis {SRC_DIR}")
    print(f"Détails : {e}")
    sys.exit(1)

# Chemin du modèle
# Remplace l'ancienne ligne MODEL_PATH par celle-là :
MODEL_PATH = os.path.join(MODELS_DIR, 'XGBoost_model.pkl')

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def model():
    """Charge le modèle XGBoost de manière sécurisée."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Fichier modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="module")
def predictions(model):
    return model.predict(X_test_final)

@pytest.fixture(scope="module")
def probabilities(model):
    return model.predict_proba(X_test_final)[:, 1]

# =============================================================================
# CLASSES DE TESTS
# =============================================================================

class TestIntegritéDonnees:
    """Vérifie que les données fournies au modèle sont saines."""

    def test_absence_nan(self):
        assert X_test_final.isna().sum().sum() == 0, "Le dataset de test contient des valeurs manquantes"

    def test_dimensions_colonnes(self, model):
        assert X_test_final.shape[1] == model.n_features_in_, \
            f"Mismatch : {X_test_final.shape[1]} features en entrée vs {model.n_features_in_} attendues."

class TestPerformanceXGBoost:
    """Vérifie que le modèle atteint les exigences de précision clinique."""

    def test_accuracy_seuil(self, model):
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, model.predict(X_test_final))
        assert acc >= 0.90, f"Performance insuffisante : {acc:.2%}"

    def test_rappel_clinique(self, model):
        """Le rappel est crucial pour ne pas rater de cas positifs."""
        from sklearn.metrics import recall_score
        rec = recall_score(y_test, model.predict(X_test_final))
        assert rec >= 0.30, f"Risque de faux négatifs trop élevé (Recall: {rec:.2%})"

class TestCoherenceClinique:
    """Vérifie que les prédictions font sens médicalement."""

    def test_haute_probabilite(self, model, probabilities, predictions):
        """Si proba > 95%, la prédiction doit être 1."""
        masque = probabilities > 0.95
        if masque.any():
            assert (predictions[masque] == 1).all(), "Incohérence : forte probabilité mais prédiction négative"

    def test_prevalence_predite(self, predictions):
        """Vérifie que le modèle ne prédit pas 'malade' pour tout le monde."""
        ratio_positif = predictions.mean()
        assert 0.03 < ratio_positif < 0.60, f"Distribution suspecte des prédictions : {ratio_positif:.2%}"

# =============================================================================
# POINT D'ENTRÉE (Pour lancer directement le script si besoin)
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])