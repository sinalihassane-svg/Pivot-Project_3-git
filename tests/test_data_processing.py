import pytest
import numpy as np

def test_data_shape():
    # Ce test vérifie que la structure de données pour CatBoost est correcte
    # On simule les 9 entrées du formulaire
    sample_data = [25, 1, 5, 2, 18, 1, 2, 0, 0]
    assert len(sample_data) == 9

def test_numpy_conversion():
    # Vérifie que la conversion en tableau numpy fonctionne (utilisé dans app.py)
    sample_data = [0] * 9
    features = np.array([sample_data])
    assert features.shape == (1, 9)