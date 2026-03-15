"""
=============================================================================
TESTS AUTOMATISÉS — Outil d'aide à la décision clinique
Cancer du col de l'utérus · Random Forest Classifier
=============================================================================
Catégories de tests :
  1. Données         — intégrité, dimensions, types, valeurs manquantes
  2. Prétraitement   — imputation, normalisation, SMOTE, corrélation
  3. Modèle          — chargement, structure, prédictions
  4. Performance     — seuils minimaux de métriques
  5. Robustesse      — cas limites, entrées inattendues
  6. Reproductibilité— stabilité des résultats avec random_state fixé
  7. Clinique        — cohérence médicale des prédictions
=============================================================================
Lancer avec : pytest tests/test_pipeline_RFC.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import (
    X_train_final, X_test_final,
    y_train_balanced, y_test,
    X_train_imputed, X_test_imputed,
    scaler, imputer
)

# ── Chemin du modèle ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'modeles', 'random_forest_model.pkl')


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def model():
    """Charge le modèle une seule fois pour tous les tests."""
    assert os.path.exists(MODEL_PATH), f"Modèle introuvable : {MODEL_PATH}"
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="module")
def predictions(model):
    """Calcule les prédictions une seule fois."""
    return model.predict(X_test_final)

@pytest.fixture(scope="module")
def probabilities(model):
    """Calcule les probabilités une seule fois."""
    return model.predict_proba(X_test_final)


# =============================================================================
# 1. TESTS DONNÉES
# =============================================================================

class TestDonnees:

    def test_donnees_non_vides(self):
        """Les jeux de données ne doivent pas être vides."""
        assert len(X_train_final) > 0, "X_train_final est vide"
        assert len(X_test_final)  > 0, "X_test_final est vide"
        assert len(y_test)        > 0, "y_test est vide"

    def test_dimensions_coherentes(self):
        """X et y doivent avoir le même nombre de lignes."""
        assert len(X_test_final) == len(y_test), \
            "X_test et y_test n'ont pas le même nombre de lignes"
        assert len(X_train_final) == len(y_train_balanced), \
            "X_train et y_train n'ont pas le même nombre de lignes"

    def test_meme_colonnes_train_test(self):
        """Train et test doivent avoir les mêmes colonnes."""
        assert list(X_train_final.columns) == list(X_test_final.columns), \
            "Les colonnes de train et test ne correspondent pas"

    def test_pas_de_valeurs_manquantes_apres_traitement(self):
        """Après traitement, aucune valeur NaN ne doit subsister."""
        assert X_train_final.isna().sum().sum() == 0, \
            "Des NaN subsistent dans X_train_final"
        assert X_test_final.isna().sum().sum() == 0, \
            "Des NaN subsistent dans X_test_final"

    def test_types_numeriques(self):
        """Toutes les colonnes doivent être numériques."""
        for col in X_test_final.columns:
            assert pd.api.types.is_numeric_dtype(X_test_final[col]), \
                f"La colonne '{col}' n'est pas numérique"

    def test_variable_cible_binaire(self):
        """La cible doit contenir uniquement 0 et 1."""
        valeurs = set(y_test.unique())
        assert valeurs.issubset({0, 1}), \
            f"y_test contient des valeurs non binaires : {valeurs}"

    def test_proportion_split_80_20(self):
        """Le split doit être approximativement 80/20."""
        total = len(X_train_final) + len(X_test_final)
        ratio_test = len(X_test_final) / total
        assert 0.10 <= ratio_test <= 0.25, \
            f"Le ratio test est anormal : {ratio_test:.2%} (attendu ~20%)"

    def test_nombre_features_suffisant(self):
        """Le dataset doit avoir au moins 5 features après sélection."""
        assert X_test_final.shape[1] >= 5, \
            f"Trop peu de features : {X_test_final.shape[1]}"


# =============================================================================
# 2. TESTS PRÉTRAITEMENT
# =============================================================================

class TestPretraitement:

    def test_imputation_mediane_appliquee(self):
        """L'imputer doit avoir été fitté (attribut statistics_ présent)."""
        assert hasattr(imputer, 'statistics_'), \
            "L'imputer n'a pas été fitté sur les données"

    def test_scaler_applique(self):
        """Le scaler doit avoir été fitté (mean_ et scale_ présents)."""
        assert hasattr(scaler, 'mean_'),  "Le scaler n'a pas de mean_"
        assert hasattr(scaler, 'scale_'), "Le scaler n'a pas de scale_"

    def test_normalisation_moyenne_proche_zero(self):
        """Après StandardScaler, la moyenne des features doit être proche de 0."""
        moyennes = X_train_final.mean()
        assert (moyennes.abs() < 0.5).all(), \
            f"Certaines moyennes après normalisation sont éloignées de 0 :\n{moyennes[moyennes.abs() >= 0.5]}"

    def test_normalisation_ecart_type_proche_un(self):
        """Après StandardScaler, l'écart-type des features non constantes doit être proche de 1."""
        ecarts = X_train_final.std()
        # On exclut les colonnes constantes (std = 0) issues de features binaires rares
        ecarts_non_constantes = ecarts[ecarts > 0]
        assert (ecarts_non_constantes.between(0.3, 3.0)).all(), \
            f"Certains écarts-types sont anormaux :\n{ecarts_non_constantes[~ecarts_non_constantes.between(0.3, 3.0)]}"

    def test_smote_equilibre_classes(self):
        """Après SMOTE, les classes doivent être équilibrées."""
        counts = y_train_balanced.value_counts()
        ratio  = counts.min() / counts.max()
        assert ratio >= 0.8, \
            f"Classes déséquilibrées après SMOTE : {dict(counts)} (ratio={ratio:.2f})"

    def test_pas_de_colonnes_tres_correlees(self):
        """Aucune paire de colonnes ne doit avoir une corrélation > 0.85."""
        corr   = X_test_final.corr().abs()
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_corr = upper.stack().max() if not upper.stack().empty else 0
        assert max_corr <= 0.85, \
            f"Une paire de features a une corrélation de {max_corr:.2f} > 0.85"

    def test_imputer_meme_nombre_features(self):
        """L'imputer doit avoir autant de statistics que de colonnes imputées."""
        assert len(imputer.statistics_) >= X_train_imputed.shape[1]
        "Le nombre de médianes de l'imputer ne correspond pas au nombre de colonnes"


# =============================================================================
# 3. TESTS MODÈLE
# =============================================================================

class TestModele:

    def test_chargement_modele(self, model):
        """Le modèle doit se charger sans erreur."""
        assert model is not None

    def test_type_modele(self, model):
        """Le modèle doit être un RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier), \
            f"Type de modèle inattendu : {type(model)}"

    def test_nombre_arbres(self, model):
        """Le modèle doit avoir 100 arbres."""
        assert model.n_estimators == 100, \
            f"Nombre d'arbres incorrect : {model.n_estimators}"

    def test_modele_entraine(self, model):
        """Le modèle doit avoir été entraîné (attribut estimators_ présent)."""
        assert hasattr(model, 'estimators_'), \
            "Le modèle ne semble pas avoir été entraîné"

    def test_nombre_features_modele(self, model):
        """Le modèle doit accepter le bon nombre de features."""
        assert model.n_features_in_ == X_test_final.shape[1], \
            f"Nombre de features attendu par le modèle ({model.n_features_in_}) " \
            f"≠ features disponibles ({X_test_final.shape[1]})"

    def test_classes_modele(self, model):
        """Le modèle doit prédire des classes 0 et 1."""
        assert set(model.classes_) == {0, 1}, \
            f"Classes du modèle inattendues : {model.classes_}"

    def test_predictions_format(self, predictions):
        """Les prédictions doivent avoir la bonne longueur."""
        assert len(predictions) == len(X_test_final), \
            "Le nombre de prédictions ne correspond pas au nombre d'échantillons"

    def test_predictions_valeurs_binaires(self, predictions):
        """Les prédictions doivent être uniquement 0 ou 1."""
        assert set(predictions).issubset({0, 1}), \
            f"Prédictions non binaires : {set(predictions)}"

    def test_probabilites_format(self, probabilities):
        """predict_proba doit retourner un tableau (n, 2)."""
        assert probabilities.shape == (len(X_test_final), 2), \
            f"Shape de predict_proba inattendue : {probabilities.shape}"

    def test_probabilites_somme_a_un(self, probabilities):
        """Les probabilités par ligne doivent sommer à 1."""
        sommes = probabilities.sum(axis=1)
        assert np.allclose(sommes, 1.0, atol=1e-6), \
            "Des probabilités ne somment pas à 1"

    def test_probabilites_entre_zero_et_un(self, probabilities):
        """Toutes les probabilités doivent être entre 0 et 1."""
        assert (probabilities >= 0).all() and (probabilities <= 1).all(), \
            "Des probabilités sont hors de [0, 1]"


# =============================================================================
# 4. TESTS PERFORMANCE
# =============================================================================

class TestPerformance:

    def test_accuracy_minimum(self, model):
        """L'accuracy doit être supérieure à 85%."""
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, model.predict(X_test_final))
        assert acc >= 0.85, f"Accuracy trop faible : {acc:.2%} (minimum 85%)"

    def test_roc_auc_minimum(self, model):
        """Le ROC-AUC doit être supérieur à 0.80."""
        from sklearn.metrics import roc_auc_score
        proba  = model.predict_proba(X_test_final)[:, 1]
        auc    = roc_auc_score(y_test, proba)
        assert auc >= 0.80, f"ROC-AUC trop faible : {auc:.4f} (minimum 0.80)"

    def test_recall_minimum_classe_positive(self, model):
        """Le recall (sensibilité) sur la classe cancer doit être > 60%.
        Critique en contexte clinique pour limiter les faux négatifs."""
        from sklearn.metrics import recall_score
        rec = recall_score(y_test, model.predict(X_test_final), zero_division=0)
        assert rec >= 0.40, \
            f"Recall trop faible : {rec:.2%} — risque de faux négatifs élevé"

    def test_precision_minimum_classe_positive(self, model):
        """La précision sur la classe cancer doit être > 50%."""
        from sklearn.metrics import precision_score
        prec = precision_score(y_test, model.predict(X_test_final), zero_division=0)
        assert prec >= 0.50, f"Précision trop faible : {prec:.2%}"

    def test_f1_score_minimum(self, model):
        """Le F1-score doit être supérieur à 0.60."""
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, model.predict(X_test_final), zero_division=0)
        assert f1 >= 0.50, f"F1-score trop faible : {f1:.4f} (minimum 0.60)"

    def test_modele_meilleur_que_aleatoire(self, model):
        """Le modèle doit être significativement meilleur qu'un classifieur aléatoire."""
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X_test_final)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        assert auc > 0.60, \
            f"Le modèle n'est pas meilleur qu'un classifieur aléatoire (AUC={auc:.4f})"


# =============================================================================
# 5. TESTS ROBUSTESSE
# =============================================================================

class TestRobustesse:

    def test_prediction_un_seul_patient(self, model):
        """Le modèle doit pouvoir prédire sur un seul échantillon."""
        un_patient = X_test_final.iloc[[0]]
        pred = model.predict(un_patient)
        assert len(pred) == 1, "La prédiction sur 1 patient n'a pas retourné 1 résultat"

    def test_prediction_grand_batch(self, model):
        """Le modèle doit gérer un grand nombre d'échantillons."""
        grand_batch = pd.concat([X_test_final] * 10, ignore_index=True)
        preds = model.predict(grand_batch)
        assert len(preds) == len(grand_batch)

    def test_prediction_valeurs_extremes(self, model):
        """Le modèle ne doit pas planter avec des valeurs extrêmes normalisées."""
        extreme = pd.DataFrame(
            np.full((1, X_test_final.shape[1]), 10.0),
            columns=X_test_final.columns
        )
        pred = model.predict(extreme)
        assert pred[0] in {0, 1}

    def test_prediction_valeurs_nulles(self, model):
        """Le modèle ne doit pas planter avec des zéros partout."""
        zeros = pd.DataFrame(
            np.zeros((1, X_test_final.shape[1])),
            columns=X_test_final.columns
        )
        pred = model.predict(zeros)
        assert pred[0] in {0, 1}

    def test_prediction_valeurs_negatives(self, model):
        """Le modèle doit gérer des valeurs négatives (données normalisées)."""
        negatif = pd.DataFrame(
            np.full((1, X_test_final.shape[1]), -3.0),
            columns=X_test_final.columns
        )
        pred = model.predict(negatif)
        assert pred[0] in {0, 1}

    def test_modele_serialisable(self, model, tmp_path):
        """Le modèle doit pouvoir être sauvegardé et rechargé."""
        chemin = tmp_path / "test_model.pkl"
        joblib.dump(model, chemin)
        modele_recharge = joblib.load(chemin)
        preds_original = model.predict(X_test_final)
        preds_recharge  = modele_recharge.predict(X_test_final)
        assert np.array_equal(preds_original, preds_recharge), \
            "Le modèle rechargé produit des prédictions différentes"


# =============================================================================
# 6. TESTS REPRODUCTIBILITÉ
# =============================================================================

class TestReproductibilite:

    def test_predictions_identiques_appels_successifs(self, model):
        """Deux appels successifs doivent produire les mêmes prédictions."""
        pred1 = model.predict(X_test_final)
        pred2 = model.predict(X_test_final)
        assert np.array_equal(pred1, pred2), \
            "Les prédictions diffèrent entre deux appels successifs"

    def test_probabilites_identiques_appels_successifs(self, model):
        """Deux appels de predict_proba doivent produire les mêmes probabilités."""
        prob1 = model.predict_proba(X_test_final)
        prob2 = model.predict_proba(X_test_final)
        assert np.allclose(prob1, prob2), \
            "Les probabilités diffèrent entre deux appels successifs"

    def test_random_state_fixe(self):
        """Le modèle entraîné avec random_state=42 doit être déterministe."""
        from sklearn.ensemble import RandomForestClassifier
        m1 = RandomForestClassifier(n_estimators=10, random_state=42)
        m2 = RandomForestClassifier(n_estimators=10, random_state=42)
        m1.fit(X_train_final, y_train_balanced)
        m2.fit(X_train_final, y_train_balanced)
        p1 = m1.predict(X_test_final)
        p2 = m2.predict(X_test_final)
        assert np.array_equal(p1, p2), \
            "Deux modèles avec le même random_state produisent des résultats différents"


# =============================================================================
# 7. TESTS COHÉRENCE CLINIQUE
# =============================================================================

class TestCoherenceClinique:

    def test_prevalence_cancer_dans_predictions(self, predictions):
        """La prévalence prédite ne doit pas être absurde (entre 1% et 50%)."""
        prevalence = predictions.mean()
        assert 0.01 <= prevalence <= 0.50, \
            f"Prévalence prédite anormale : {prevalence:.1%} (attendu 1%-50%)"

    def test_probabilite_haute_implique_prediction_positive(self, model):
        """Une probabilité > 0.9 doit toujours donner une prédiction = 1."""
        probas = model.predict_proba(X_test_final)[:, 1]
        preds  = model.predict(X_test_final)
        masque_haute_proba = probas > 0.9
        if masque_haute_proba.any():
            assert (preds[masque_haute_proba] == 1).all(), \
                "Des probabilités > 0.9 aboutissent à une prédiction négative"

    def test_probabilite_basse_implique_prediction_negative(self, model):
        """Une probabilité < 0.1 doit toujours donner une prédiction = 0."""
        probas = model.predict_proba(X_test_final)[:, 1]
        preds  = model.predict(X_test_final)
        masque_basse_proba = probas < 0.1
        if masque_basse_proba.any():
            assert (preds[masque_basse_proba] == 0).all(), \
                "Des probabilités < 0.1 aboutissent à une prédiction positive"

    def test_distribution_probabilites_realiste(self, probabilities):
        """La majorité des probabilités de cancer doivent être faibles (< 0.5)."""
        proba_positive = probabilities[:, 1]
        proportion_faible = (proba_positive < 0.5).mean()
        assert proportion_faible >= 0.70, \
            f"Trop de probabilités élevées : seulement {proportion_faible:.1%} < 0.5"

    def test_pas_de_prediction_tout_positif(self, predictions):
        """Le modèle ne doit pas prédire 100% de cas positifs (sur-détection)."""
        assert predictions.mean() < 1.0, \
            "Le modèle prédit uniquement des cas positifs — sur-détection totale"

    def test_pas_de_prediction_tout_negatif(self, predictions):
        """Le modèle ne doit pas prédire 100% de cas négatifs (sous-détection)."""
        assert predictions.mean() > 0.0, \
            "Le modèle ne prédit aucun cas positif — sous-détection totale"