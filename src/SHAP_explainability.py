import shap
import pandas as pd
import joblib  # Module standard pour charger/sauvegarder les modèles
import matplotlib.pyplot as plt

def shap_explainability(modele, X_patient):
    """
    Génère l'explicabilité SHAP pour les modèles basés sur des arbres 
    (Random Forest, XGBoost, CatBoost) en gérant leurs différences de format.
    
    Paramètres:
    - modele : Le modèle d'apprentissage automatique déjà entraîné.
    - X_patient : Un DataFrame (1 ligne) contenant les données prétraitées de la patiente.
    
    Retourne:
    - explainer : L'objet TreeExplainer de SHAP.
    - shap_values_cible : Les valeurs SHAP associées uniquement à la classe "À risque" (1).
    """
    # Initialisation de l'explicateur SHAP spécifique aux arbres
    explainer = shap.TreeExplainer(modele)
    
    # Calcul des valeurs SHAP pour la patiente
    shap_values = explainer.shap_values(X_patient)
    
    # Gestion de la différence de format de sortie entre les bibliothèques
    if isinstance(shap_values, list):
        # Pour Random Forest (Scikit-Learn) : on récupère les valeurs de la classe 1
        shap_values_cible = shap_values[1]
    else:
        # Pour XGBoost et CatBoost : le format est déjà correct pour la classe positive
        shap_values_cible = shap_values
        
    return explainer, shap_values_cible

# --- LIGNES DE TEST AJOUTÉES ---
if __name__ == "__main__":
    # 1. Importation des données de test
    # On importe X_test_final depuis ton fichier data_processing.py
    # (Assure-toi que data_processing.py ne contient pas d'erreurs, car cet import va l'exécuter)
    from data_processing import X_test_final
    
    # On sélectionne la première patiente du jeu de test (index 0)
    # Le double crochet [[0]] ou l'utilisation de .iloc[[0]] garantit que X_patient reste un DataFrame 2D (requis par SHAP)
    X_patient_test = X_test_final.iloc[[0]]

    # 2. Chargement des modèles entraînés
    # Remplace les noms de fichiers par les chemins exacts où tu as sauvegardé tes modèles
    try:
        modeles_a_tester = {
            "Random Forest": joblib.load('modele_rfc.pkl'),
            "XGBoost": joblib.load('modele_xgboost.pkl'),
            "CatBoost": joblib.load('modele_catboost.pkl')
        }

        # 3. Boucle de test pour générer les graphiques SHAP de chaque modèle
        for nom_modele, modele in modeles_a_tester.items():
            print(f"Génération de l'explicabilité pour {nom_modele}...")
            
            # Appel de ta fonction
            explainer, shap_values_cible = shap_explainability(modele, X_patient_test)
            
            # Ajustement de la valeur de base selon le format renvoyé par l'explainer
            valeur_de_base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            
            # Formatage pour le graphique Waterfall
            explication = shap.Explanation(
                values=shap_values_cible[0], 
                base_values=valeur_de_base, 
                data=X_patient_test.iloc[0].values, 
                feature_names=X_patient_test.columns.tolist()
            )
            
            # Affichage du graphique
            plt.figure(figsize=(10, 6))
            # show=False permet de personnaliser le graphique avant de l'afficher avec plt.show()
            shap.plots.waterfall(explication, show=False)
            plt.title(f"Impact des facteurs de risque - Modèle : {nom_modele}")
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError as e:
        print(f"\nERREUR : Impossible de trouver un fichier de modèle. {e}")
        print("Assure-toi d'avoir ajouté 'joblib.dump(modele, \"nom_du_fichier.pkl\")' à la fin de tes scripts d'entraînement.")