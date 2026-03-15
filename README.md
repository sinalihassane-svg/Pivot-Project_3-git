# 🩺 Application d'Aide à la Décision Médicale - Évaluation du Risque de Cancer du Col de l'Utérus

Ce projet est réalisé dans le cadre de la Coding Week. Il s'agit d'un outil d'aide à la décision clinique permettant d'évaluer le risque de cancer du col de l'utérus chez les patientes en fonction de leurs antécédents médicaux et de leurs facteurs comportementaux. Notre outil se base sur des modèles de Machine Learning (Random Forest, XGBoost, CatBoost), avec une explicabilité assurée par SHAP.

**Équipe :** 
- Bakayoko Mouhamed Soualiou(BakMomoS)
- Diallo Ismaila(dialloismaila256messi-gif)
- Gbatta Jovite Jean-Paul(jovitejeanpaul)
- Mounirou Kouadio Kobenan Habib(Mounirou-H-ops)
- Ouattara El Hadj Sinali(sinalihassane-svg)

**Objectifs du projet :**
- Développer un modèle d'apprentissage automatique robuste et explicable.
- Garantir la transparence des prédictions du modèle grâce à l'explicabilité SHAP.
- Créer une interface utilisateur intuitive (Streamlit ou Flask).
- Suivre les bonnes pratiques de développement logiciel (GitHub, CI/CD automatisée).
- Faire preuve de réactivité en documentant clairement les invites générées par l'IA utilisées dans votre flux de travail.


```text
Pivot-Project_3-git
|
├── .github/             # Configuration pour le pipeline CI/CD (GitHub Actions)
├── app/                 # Interface utilisateur
├── catboost_info/       # Logs et informations générés automatiquement par CatBoost
├── data/                # Jeu de données brut (CSV)
├── images/              # Visualisations sauvegardées (proportions, corrélations, graphes SHAP)
├── modèles/             # Modèles d'apprentissage automatique sauvegardés (.pkl)
├── notebooks/           # Analyse exploratoire des données (EDA)
├── reports/             # Rapports générés
├── src/                 # Code source (prétraitement, entraînement, explicabilité)
├── tests/               # Scripts de tests unitaires
├── .gitattributes       # Configuration des attributs Git
├── .gitignore           # Fichiers et dossiers à ignorer par Git
├── Dockerfile           # Configuration pour la conteneurisation de l'application
├── README.md            # Ce document de présentation
└── requirements.txt     # Liste des dépendances Python nécessaires au projet
```


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

**Modèle CatBoost Classifier**
```bash
python src/train_model_CATBoost.py

```
**Modèle Random Forest Classifier**
```bash
python src/train_model_RFC.py

```
**Modèle XGBoost Classifier**
```bash
python src/train_model_XGBoost.py

```

**4. Lancer l'application web :**

```bash
python app/app.py

```

## 📊 Traitement des données

### Gestion des valeurs manquantes
Pour la gestion des valeurs manquantes, nous avons commencé par supprimer les caractéristiques (colonnes) avec une proportion de valeurs manquantes supérieure ou égale à 60%.
Nous avons aussi remarqué que certaines des patientes ont très peu de données. Nous avons donc décidé de supprimer de la base de données les patients avec une proportion de valeurs manquantes supérieure ou égale à 60%.

### Gestion des valeurs aberrantes
Une valeur aberrante est une observation qui s'éloigne de façon anormale ou extrême des autres valeurs de ta base de données. Elle semble complètement "hors norme" par rapport au comportement général de tes échantillons.
Pour la détection des valeurs aberrantes, on utilise la méthode IQR(Interquartile Range). Après les avoir détectées, on les supprime de la base de données.

### Remplacement des valeurs manquantes
Enfin nous remplaçons les valeurs manquantes restantes après la suppression de certaines colonnes et lignes et celles créees par la suppression des valeurs aberrantes par la médiane de chacune des caractéristiques, calculée uniquement sur la base de données d'entrainement pour éviter un data leakage (si on l'avait calculée sur toute la base de données)

### Gestion du déséquilibre
Le jeu de données initial présentait un déséquilibre majeur (94,8% de cas "Sans risque" contre 5,2% "À risque"). Nous avons appliqué la méthode **SMOTE (Synthetic Minority Over-sampling Technique)** uniquement sur les données d'entraînement.
**Impact :** Cela a permis de générer des exemples synthétiques pour la classe minoritaire, évitant au modèle de toujours prédire la classe majoritaire et améliorant considérablement sa sensibilité pour la détection des cas à risque.

### Calcul de la matrice de corrélation
La matrice de corrélation nous donne à quel point deux caractéristiques différentes donnent la même information. Dans le cas, où deux caractéristiques sont fortement corrélées, on en garde qu'une. Voici la liste des caractéristiques que nous avons supprimées:
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


## 🔥 Performance des modèles
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
3. Le temps de tabagisme (Smokes (years))


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
3. L'âge 


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

Avant la réalisation de ce projet, nos compétences en Prompt Engineering n'étaient pas mauvaise mais elles n'étaient pas au niveau que nous avons aujourd'hui. Nous avions très souvent du mal à obtenir les réponses que nous attendions de l'IA.
La Coding Week nous a réellement permis de développer nos compétences en Prompt Enginnering en un laps de temps très court.
- Nous avons élargi notre culture scientifique sur les modèles d'IA (LM, LLM, GenIA,..)
- Nous comprenons mieux comment structurer nos prompt (Contexte-Tâche-Contrainte)
- Nous avons une meilleure connaissance sur la notion de cycle d'itération du prompt
- Aussi, sachant que les IA cherchent toujours à nous donner une reponse même quand cette reponse n'existe pas (hallucination), nous sommes beaucoup plus attentifs à leurs réponses.

Tout au long de ce projet, nous ne nous sommes pas contentés de prompter, nous avons aussi cherché à comprendre les codes que nous obtenions.
Cela nous a permis de développer une compréhension plus fine sur le Machine Learning notamment:
- L'utilisation de GitHub 
- L'optimisation de la mémoire
- L'exploration des données
- Le traitement des données (division de la base de données, valeurs manquantes, valeurs aberrantes, déséquilibre, corrélation, data leakage)
- La compréhension des modèles ( CatBoost Classifier, XGBoost Classifier, Random Forest Classifier, SVM)
- L'entrainement des modèles
- L'évaluation des modèles (les indicateurs de performance ROC-AUC, accuracy, precision, recall et F1-score)
- L'explicabilité SHAP (Prise de décision des modèles)
- La création de l'interface web (Front-end, Back-end)
- L'intégration et le Développement Continus (GitHub Actions)