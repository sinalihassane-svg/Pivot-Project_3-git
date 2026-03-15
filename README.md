# 🩺 Application d'Aide à la Décision Médicale - Évaluation du Risque de Cancer du Col de l'Utérus

Ce projet est réalisé dans le cadre de la Coding Week. Il s'agit d'un outil d'aide à la décision clinique permettant d'évaluer le risque de cancer du col de l'utérus chez les patientes en fonction de leurs antécédents médicaux et de leurs facteurs comportementaux. Notre outil se base des modèles de Machine Learning (Random Forest, XGBoost, CatBoost), avec une explicabilité assurée par SHAP.

**Équipe :** 
- Bakayoko Mouhamed Soualiou(BakMomoS)
- Diallo Ismaila(dialloismaila256messi-gif)
- Gbatta Jovite Jean-Paul(jovitejeanpaul)
- Mounirou Kouadio Kobenan Habib(Mounirou-H-ops)
- Ouattara El Hadj Sinali(sinalihassane-svg)

**Objectifs du projet :**
- Développez un modèle d'apprentissage automatique robuste et explicable.
- Garantissez la transparence des prédictions du modèle grâce à l'explicabilité SHAP.
- Créez une interface utilisateur intuitive (Streamlit ou Flask).
- Suivez les bonnes pratiques de développement logiciel (GitHub, CI/CD automatisée).
- Faites preuve de réactivité en documentant clairement les invites générées par l'IA utilisées dans votre flux de travail.


## 📂 Structure du Répertoire

* `app/` : Interface utilisateur Streamlit (`app.py`).
* `src/` : Scripts de prétraitement, d'entraînement et d'explicabilité SHAP.
* `notebooks/` : Analyse exploratoire (EDA).
* `images/` : Visualisations et graphiques SHAP sauvegardés automatiquement.
* `.github/workflows/` : Pipeline CI/CD.


## ⚙️ Reproductibilité : Installation et Exécution

Pour lancer l'entraînement des modèles et démarrer l'interface web localement, suivez ces étapes :

**1. Installer les dépendances :**
```bash
pip install -r requirements.txt

```

**2. Lancer les tests :**
```bash
pytest tests/

```

**3. Entraîner les modèles de Machine Learning :**

```bash
python src/train_model_CATBoost.py
python src/train_model_RFC.py
python src/train_model_XGBoost.py

```

**4. Lancer l'application web :**

```bash
python app/app.py

```

## 📊 Traitement des données

### Gestion des valeurs manquantes
Pour la gestion des valeurs manquantes, nous avons commencer par supprimer les caractéristiques (colonnes) avec une proportion de valeurs manquantes supérieur ou égale à 60%.
Nous avons aussi remarqué que certaines des patientes ont très peu de données. Nous avons donc décidé de supprimer de la base de données les patients avec une proportion de valeurs manquantes supérieur ou égale à 60%.

### Gestion des valeurs aberrantes
Une valeur aberrante est une observation qui s'éloigne de façon anormale ou extrême des autres valeurs de ta base de données. Elle semble complètement "hors norme" par rapport au comportement général de tes échantillons.
Pour la détection des valeurs aberrantes, on utilise la méthode IQR(Interquartile Range). Après les avoir détecter, on les supprime de la base de données.

### Remplacement des valeurs manquantes
Enfin nous remplaçons les valeurs manquantes restantes après la suppression de certaines colonnes et lignes et celle crées par la suppression des valeurs aberrantes par la médiane de chacune des caractéristiques, calculée uniquement sur la base de données d'entrainement pour éviter un data leakage (si on l'avait calculée sur toute la base de données)

### Gestion du déséquilibre
Le jeu de données initial présentait un déséquilibre majeur (94,8% de cas "Sans risque" contre 5,2% "À risque"). Nous avons appliqué la méthode **SMOTE (Synthetic Minority Over-sampling Technique)** uniquement sur les données d'entraînement.
**Impact :** Cela a permis de générer des exemples synthétiques pour la classe minoritaire, évitant au modèle de toujours prédire la classe majoritaire et améliorant considérablement sa sensibilité pour la détection des cas à risque.

### Calcul de la matrice de corrélation
La matrice de corrélation nous donne à quel point deux caractéristiques différentes donne la même information. Dans le cas, où deux caractéristiques sont fortement corrélées, on en garde qu'une. Voici la liste des caractéristiques que nous avons supprimer:
- STDs (number)
- STDs: condylomatosis
- STDs:vulvo-perineal condylomatosis
- STDs: Number of diagnosis
- Dx:HPV


## 📋 Liste des caractéristiques utilisées
- Age
- Number of sexual partners
- First sexual intercourse
- Number of pregnancies
- Smokes
- Smoke (years)
- Smoke (packs/year)
- Hormonal Contraceptives
- Hormonal Contraceptives(years)
- IUD 
- IUD (years)
- STDs
- STDs:cervical condylomatosis
- STDs:vaginal condylomatosis
- STDs:syphilis
- STDs:pelvic inflammatory disease
- STDs:genital herpes
- STDs:molluscum contagiosum
- STDs:AIDS
- STDs:HIV
- STDs:Hepatitis B
- STDs:HPV
- Dx:Cancer
- Dx:CIN
- Dx
- Hinselmann
- Schiller
- Citology


## 🔥 Perfomance des modèles
### Modèle CatBoost Classifier
#### Performances
* Accuracy : 96%
* Précision :98%(saine),70%(Risque)
* Rappel (Recall) pour la classe À risque : 64%
* F1-Score  Pour la classe risque: 67%
* ROC-AUC : 0,95(95%)

#### Quelles caractéristiques médicales ont le plus influencé les prédictions (Résultats SHAP) ?
L'utilisation de `TreeExplainer` de SHAP a révélé que les facteurs suivants ont le plus fort impact sur la probabilité de risque :

1. L'âge
2. Le nombre de grossesses (Num of pregnancies)
3.  Le temps de tabagisme (Smokes (years))


### Modèle XGBoost Classifier
#### Performances
* Accuracy : 94.19%
* Précision :57.14%
* Rappel (Recall) pour la classe À risque : 34.36%
* F1-Score  Pour la classe risque: 44.44%
* ROC-AUC : 0.9554(95.54%)

#### Quelles caractéristiques médicales ont le plus influencé les prédictions (Résultats SHAP) ?

L'utilisation de `TreeExplainer` de SHAP a révélé que les facteurs suivants ont le plus fort impact sur la probabilité de risque :

1. Schiller 
2. Hinselman 
3.  L'âge 


### Modèle Random Forest Classifier
#### Performances
* Accuracy : 96.8%
* Précision : 62.5%
* Rappel (Recall) pour la classe À risque : 45.5%
* F1-Score  Pour la classe risque: 52.6%
* ROC-AUC : 0.9684

#### Quelles caractéristiques médicales ont le plus influencé les prédictions (Résultats SHAP) ?

L'utilisation de `TreeExplainer` de SHAP a révélé que les facteurs suivants ont le plus fort impact sur la probabilité de risque :

1. Schiller
2. Hinselmann
3. Citoly
4. Dx
5. Hormonal contraceptive


**Parmi les modèles testés, le modèle CatBoost classifier a démontré les meilleures performances sur notre ensemble de test.**


## 💻 Quels enseignements le "Prompt Engineering" a-t-il apportés à notre projet ?

Avant la réalisation de ce projet, nos compétences en Prompt Engineering n'était pas mauvaise mais elles n'étaient pas au niveau que nous avons aujourd'hui. Nous rencontrions très souvent du mal à obtenir les reponses que nous attendions de l'IA.
La Coding Week nous a réellement permis de développement nos compétences en Prompt Enginnering en un lapse de temps très court.
- Nous avons élargir notre culture scientifique sur les modèles d'IA (LM, LLM, GenIA,..)
- Nous comprenons mieux comment structurer nos prompt (Contexte-Tâche-Contrainte)
- Nous avons une meilleur connaissance sur la notion de cycle d'itération du prompt
- Aussi, sachant que les IA cherchent toujours à nous donner une reponse même quand cette reponse n'existe pas (allucination), nous sommes beaucoup plus attentif sur leur reponse.
Tout au long de ce projet, nous ne nous sommes pas contenter de prompter, nous avons aussi chercher à comprendre les codes que nous obtenions.
Cela nous a permis de développement une compréhension plus fine sur le Machine Learning notamment:
- L'utilisation de GitHub 
- L'optimisation de la mémoire
- L'exploration des données
- Le traitement des données (division de la base de données, valeurs manquantes, valeurs aberrantes, déséquilibre, corrélation, data leakage)
- La compréhension des modèles ( CatBoost Classifier, XGBoost Classifier, Random Forest Classifier, SVM)
- L'entrainement des modèles
- L'évaluation des modèles (les indicateurs de performance ROC-AUC, accuracy, precision, recall et F1-score)
- L'explicabilité SHAP (Prise de décision des modèles)
- La création de l'interface wzb (Front-end, Back-end)
- L'intégration et le Développement Continus (GitHub Actions)