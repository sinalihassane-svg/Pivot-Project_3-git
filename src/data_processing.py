
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Récupération de la base de données via l'URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
df = pd.read_csv(url, na_values=["?"])

# Définition de la cible
X = df.drop('Biopsy', axis=1)
y = df['Biopsy']

# 2. Division en 80% entraînement et 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Retrait des colonnes avec >= 60% de valeurs manquantes
seuil_col = 0.6 * len(X_train)
colonnes_a_garder = X_train.columns[X_train.isna().sum() < seuil_col]
X_train = X_train[colonnes_a_garder]
X_test = X_test[colonnes_a_garder]

# 4. Retrait des lignes avec >= 60% de valeurs manquantes
seuil_ligne = 0.6 * len(X_train.columns)
lignes_a_garder = X_train.isna().sum(axis=1) < seuil_ligne
X_train = X_train[lignes_a_garder]
y_train = y_train[lignes_a_garder]

# 5. Détection et suppression des valeurs aberrantes (IQR) - CORRIGÉ
# On calcule l'IQR et on filtre uniquement les colonnes où IQR > 0 
# Cela évite d'éliminer les "1" dans les colonnes binaires (0/1)
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1

colonnes_a_filtrer = IQR[IQR > 0].index
condition_iqr = ~((X_train[colonnes_a_filtrer] < (Q1[colonnes_a_filtrer] - 1.5 * IQR[colonnes_a_filtrer])) | \
                  (X_train[colonnes_a_filtrer] > (Q3[colonnes_a_filtrer] + 1.5 * IQR[colonnes_a_filtrer]))).any(axis=1)

X_train = X_train[condition_iqr]
y_train = y_train[condition_iqr]

# 6. Remplacement des valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# 7. Matrice de corrélation
corr_matrix = X_train_imputed.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
colonnes_a_supprimer = [col for col in upper.columns if any(upper[col] > 0.85)]

X_train_imputed = X_train_imputed.drop(columns=colonnes_a_supprimer)
X_test_imputed = X_test_imputed.drop(columns=colonnes_a_supprimer)

# 8. Gestion du déséquilibre avec SMOTE - SÉCURISÉ
# On s'assure que le nombre de voisins (k_neighbors) ne dépasse pas le nombre d'échantillons disponibles
n_samples_minority = y_train.value_counts().min()
k_neighbors = min(5, n_samples_minority - 1)

if k_neighbors > 0:
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)
else:
    X_train_balanced, y_train_balanced = X_train_imputed, y_train

# 9. Normalisation des valeurs
scaler = StandardScaler()
X_train_final = pd.DataFrame(scaler.fit_transform(X_train_balanced), columns=X_train_imputed.columns)
X_test_final = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)



# --- AJOUT DES VISUALISATIONS (SAUVEGARDE EN PNG) ---
import os

# Définition du chemin vers le dossier "images" existant à la racine
# __file__ pointe sur src/data_processing.py, donc on remonte d'un cran
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_racine = os.path.dirname(dossier_actuel)
dossier_images = os.path.join(dossier_racine, "images")

# 10. Visualisation des proportions de classes (Avant / Après SMOTE)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Graphique 1 : Avant la gestion du déséquilibre
axes[0].pie(y_train.value_counts(), labels=["Sans risque (0)", "À risque (1)"], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
axes[0].set_title("Proportion des classes AVANT SMOTE")

# Graphique 2 : Après la gestion du déséquilibre
axes[1].pie(y_train_balanced.value_counts(), labels=["Sans risque (0)", "À risque (1)"], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
axes[1].set_title("Proportion des classes APRÈS SMOTE")

plt.tight_layout()

# Sauvegarde directe dans le dossier images
chemin_pie = os.path.join(dossier_images, "proportion_classes_smote.png")
plt.savefig(chemin_pie, bbox_inches='tight', dpi=300)
plt.close() # Libère la mémoire
print(f"Graphique des proportions sauvegardé ici : {chemin_pie}")


# 11. Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(X_train_imputed.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Matrice de corrélation des caractéristiques")

# Sauvegarde directe dans le dossier images
chemin_heatmap = os.path.join(dossier_images, "matrice_correlation.png")
plt.savefig(chemin_heatmap, bbox_inches='tight', dpi=300)
plt.close() # Libère la mémoire
print(f"Graphique de corrélation sauvegardé ici : {chemin_heatmap}")
# %%
