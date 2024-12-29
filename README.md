# projet-Python
# README - Analyse de Données et Modélisation Machine Learning

## Description du projet
Ce projet vise à répondre à la problématique suivante : peut-on prédire la prévalence des maladies respiratoires en fonction du département, de la classe d'âge, du sexe, du niveau prioritaire et des niveaux d'O₃ ? Pour cela, nous avons analysé les données relatives à la qualité de l'air et aux maladies respiratoires dans la région parisienne, en réalisant une exploration approfondie des données, un prétraitement (incluant le nettoyage et l'encodage des variables), ainsi des modélisations prédictives .

## Structure du projet
Le projet est divisé en plusieurs étapes, présentées dans différents fichiers de notebook ou scripts Python.

### 1. Analyse exploratoire 
- Chargement et inspection des données (fichiers CSV relatifs à la qualité de l'air et aux pathologies).
- Visualisation des distributions et détection des valeurs manquantes.
- Transformation et agrégation des données par département et par année.
- Analyse descriptive initiale des variables (distributions, statistiques descriptives).
- Analyse descriptive post-prétraitement pour valider la cohérence des transformations.

### 2. Prétraitement des données 
- Traitement des valeurs manquantes par imputation ou suppression.
- Encodage des variables qualitatives (One-Hot Encoding pour les départements et les tranches d'âge, transformation binaire pour le sexe).
- Fusion des différentes sources de données en un dataset unifié (principal_data4).

### 3. Modélisation 
- Séparation des données en ensembles d'entraînement et de test.
- Entraînement d’un modèle Random Forest pour prédire la prévalence des maladies respiratoires (à partir des indices de qualité de l'air et des autres variables).
- Évaluation du modèle (évaluation de la performance avec le score R² pour la régression).

### 4. Visualisation et résultats 
- Visualisation des performances du modèle sur le jeu de test.
- Analyse des features importantes dans le modèle Random Forest (Feature Importance).
- Recommandations basées sur les résultats obtenus.

## Prérequis
- Python 3.8+
- Librairies Python :
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - missingno
  - openpyxl

Pour installer les dépendances, utilisez le fichier `requirements.txt` :
```bash
pip install -r requirements.txt
```

## Données utilisées
1. **Qualité de l'air** :
   - Fichier : `departement_means.csv`
   - Contient les indices de pollution (émissions de NO2, PM10, O3) par département en Ile-de-France.

2. **Pathologies respiratoires** :
   - Fichier : `effectifs (4).csv`
   - Contient les statistiques sur les maladies respiratoires par département, tranche d’âge et sexe.

3. **Dataset final** :
   - Fichier généré : `principal_data.csv`
   - Fusion des données ci-dessus après nettoyage et transformation.

## Les Modèles utilisés
Les algorithmes suivants ont été utilisés pour prédire la variable cible prev (prévalence des maladies respiratoires) :
- `Random Forest`
- `XGBoost`
- `SVM (Support Vector Machine)`

### Paramètres du modèle
- `n_estimators=300`
- `max_depth=None`
- `max_features': 'sqrt`
- `min_samples_leaf': 1`
- `min_samples_split': 2`



## Comment exécuter le projet
1. Clonez ce dépôt.
2. Placez les fichiers de données dans le dossier `data/`.
3. Exécutez les notebooks dans l’ordre suivant :
   - `qualite_aire`
   - `analyse_exploratoire.ipynb`
   - `modélisation.ipynb`
   

4. Consultez les visualisations et les résultats .



## Licence
Ce projet est sous licence **Open Database License (ODbL)**.

## Auteur
- `EL GHAOUTH MOHAMED MAHMOUD`
- `TANGOUO KUETE Ivana`
- `MAKAMWE Pierrette Josiane`
