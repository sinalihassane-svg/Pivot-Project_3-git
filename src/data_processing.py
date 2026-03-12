
#Importation des donnée
import pandas as pd

# Chemin d'accès local
chemin = r"C:\Cycle Ingénieur\1A\S6\GitHub\cervical+cancer+risk+factors\risk_factors_cervical_cancer.csv"

# Chargement du fichier CSV
data = pd.read_csv(chemin, na_values='?')

from sklearn.model_selection import train_test_split

# Nettoyage temporaire des valeurs manquantes pour permettre l'exécution des algorithmes
# (À remplacer par votre méthode d'imputation finale)


# Séparation des caractéristiques et de la cible
X = data.drop(columns=['Biopsy'])
y = data['Biopsy']

# Division : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

from sklearn.preprocessing import RobustScaler

# Initialisation de l'outil de mise à l'échelle robuste
scaler = RobustScaler()

# Apprentissage des paramètres sur les données d'entraînement et transformation
X_train_scaled = scaler.fit_transform(X_train)

# Transformation stricte des données de test (sans réapprentissage pour éviter la fuite de données)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialisation et entraînement d'un modèle
modele = RandomForestClassifier(random_state=42, class_weight='balanced')
modele.fit(X_train_scaled, y_train)

# Génération des prédictions sur les 20% de données de test
y_pred = modele.predict(X_test_scaled)

# Création de la matrice de confusion
matrice_conf = confusion_matrix(y_test, y_pred)

# Affichage visuel de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=matrice_conf, display_labels=["Pas de risque", "À risque"])
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion sur l'ensemble de Test (20%)")
plt.show()

