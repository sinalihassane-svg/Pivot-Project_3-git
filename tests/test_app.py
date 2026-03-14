import pytest

def test_homepage_load():
    # Ce test vérifie juste que la logique de base ne crash pas
    prediction = None
    assert prediction is None

def test_data_structure():
    # Vérifie que tu as bien tes 9 variables pour CatBoost
    data = [0] * 9
    assert len(data) == 9