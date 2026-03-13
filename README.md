# Pivot-Project_3-git
 MEDICAL DECISION SUPPORT APPLICATION  CERVICAL CANCER RISK ASSESSMENT WITH EXPLAINABLE ML (SHAP)

# 🩺 Application d'Aide à la Décision Médicale - Évaluation du Risque de Cancer du Col de l'Utérus

Ce projet est réalisé dans le cadre de la Coding Week de l'École Centrale Casablanca. Il s'agit d'un outil d'aide à la décision clinique permettant d'évaluer le risque de cancer du col de l'utérus chez les patientes grâce à des modèles de Machine Learning (Random Forest, XGBoost, CatBoost), avec une explicabilité assurée par SHAP.

**Équipe :** Mouhamed, Bastien, Tawba, Abderrazak, Mohammed et Idriss.

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

## 📊 Réponses aux Questions Critiques

### Le jeu de données était-il équilibré ? Si non, comment le déséquilibre a-t-il été géré et quel en a été l'impact ?

Le jeu de données initial présentait un déséquilibre majeur (~85% de cas "Sans risque" contre ~15% "À risque"). Nous avons appliqué la méthode **SMOTE (Synthetic Minority Over-sampling Technique)** uniquement sur les données d'entraînement.
**Impact :** Cela a permis de générer des exemples synthétiques pour la classe minoritaire, évitant au modèle de toujours prédire la classe majoritaire et améliorant considérablement sa sensibilité pour la détection des cas à risque.

### Quel modèle de Machine Learning a obtenu les meilleures performances ?

Parmi les modèles testés, le modèle **[Nom du modèle, ex: XGBoost]** a démontré les meilleures performances sur notre ensemble de test.

* Précision (Accuracy) : **[XX]%**
* Rappel (Recall) pour la classe À risque : **[XX]%**
* F1-Score : **[XX]%**

### Quelles caractéristiques médicales ont le plus influencé les prédictions (Résultats SHAP) ?

L'utilisation de `TreeExplainer` de SHAP a révélé que les facteurs suivants ont le plus fort impact sur la probabilité de risque :

1. **[Ex : L'âge (Age)]**
2. **[Ex : Le nombre de grossesses (Num of pregnancies)]**
3. **[Ex : Le temps de tabagisme (Smokes (years))]**

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

```

[https://docs.streamlit.io/library/get-started/main-concepts](https://docs.streamlit.io/library/get-started/main-concepts)

```