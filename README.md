# 🩺 Application d'Aide à la Décision Médicale - Évaluation du Risque de Cancer du Col de l'Utérus

Ce projet est réalisé dans le cadre de la Coding Week. Il s'agit d'un outil d'aide à la décision clinique permettant d'évaluer le risque de cancer du col de l'utérus chez les patientes en fonction de leurs antécédents médicaux et de leurs facteurs comportementaux. Notre outil se base des modèles de Machine Learning (Random Forest, XGBoost, CatBoost), avec une explicabilité assurée par SHAP.

**Équipe :** 
- Bakayoko Mouhamed Soualiou(BakMomos)
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


## ⚙️ Reproductibilité : Installation et Exécution

Pour lancer l'entraînement des modèles et démarrer l'interface web localement, suivez ces étapes :

**1. Installer les dépendances :**
```bash
pip install -r requirements.txt

```

**2. Entraîner les modèles de Machine Learning :**

```bash
python src/train_model.py

```

**3. Lancer l'application web :**

```bash
streamlit run app/app.py

```

## 📊 Traitement des données

### Gestion des valeurs manquantes
Pour la gestion des valeurs manquantes, nous avons commencer par supprimer les caractéristiques (colonnes) avec une proportion de valeurs manquantes supérieur ou égale à 60%.
Nous avons aussi remarqué que certaines des patientes ont très peu de données. Nous avons donc décidé de supprimer de la base de données les patients avec une proportion de valeurs manquantes supérieur ou égale à 60%.

### Gestion des valeurs aberrantes
Une valeur aberrante est une observation qui s'éloigne de façon anormale ou extrême des autres valeurs de ta base de données. Elle semble complètement "hors norme" par rapport au comportement général de tes échantillons.
Pour la détection des valeurs aberrantes, on utilise 
### Gestion du déséquilibre

Le jeu de données initial présentait un déséquilibre majeur (~85% de cas "Sans risque" contre ~15% "À risque"). Nous avons appliqué la méthode **SMOTE (Synthetic Minority Over-sampling Technique)** uniquement sur les données d'entraînement.
**Impact :** Cela a permis de générer des exemples synthétiques pour la classe minoritaire, évitant au modèle de toujours prédire la classe majoritaire et améliorant considérablement sa sensibilité pour la détection des cas à risque.

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


### Perfomance des modèles
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

1. L'âge
2. Le nombre de grossesses (Num of pregnancies)
3.  Le temps de tabagisme (Smokes (years))


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
6. STDs
7. Dx : Cancer




Parmi les modèles testés, le modèle CatBoost classifier a démontré les meilleures performances sur notre ensemble de test.


### Quels enseignements le "Prompt Engineering" a-t-il apportés à cette tâche ?

Le prompt engineering itératif a permis de :

* Structurer correctement la prévention du Data Leakage (fuite de données) lors de l'imputation par la médiane.
* Écrire des fonctions robustes pour harmoniser les sorties SHAP (qui différaient entre Scikit-Learn, XGBoost et CatBoost).
* Automatiser la sauvegarde des graphiques en `.png` pour ne pas bloquer le pipeline d'intégration continue (CI/CD) GitHub Actions avec des affichages interactifs.

## 📂 Structure du Répertoire

* `app/` : Interface utilisateur Streamlit (`app.py`).
* `src/` : Scripts de prétraitement, d'entraînement et d'explicabilité SHAP.
* `notebooks/` : Analyse exploratoire (EDA).
* `images/` : Visualisations et graphiques SHAP sauvegardés automatiquement.
* `.github/workflows/` : Pipeline CI/CD.

