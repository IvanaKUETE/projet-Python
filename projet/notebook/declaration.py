
# Librairies standards
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns

import xgboost as xgb
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LassoCV,RidgeCV,Ridge,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm



import warnings
warnings.filterwarnings('ignore')


### *************  Fonctions liées à l'analyse axploratoire *************************


## 1-Fonction de nettoyege de la qualité de l'air 
"""
Cette fonction permet de faire  un resumé de la qualité de l'air dans les departement de Paris  
puisque nos informations concernant la qualité de l'air sont collectés mensuellement.
"""
def process_data(input_file, output_file):
    data = pd.read_csv(input_file, sep=",")
    departments = [0, 75, 77, 78, 91, 92, 93, 94, 95]
    data = data[data['ninsee'].isin(departments)]
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    agg_data = data.groupby('ninsee')[['pm10', 'o3', 'no2']].agg({
        'pm10': 'median',
        'o3': 'mean',
        'no2': 'median'
    }).reset_index()
    agg_data.to_csv(output_file, index=False)




### *************  Fonctions liées à la modélisation *************************

### 1-  formatage de données
"""
Ici il est question d'avoir une vu d'ensemble sur les variables presente dans notre dataset
"""
def formatage_variable(dataframe):

    # Afficher les types des variables
    print("Les types de nos variables:")
    print(dataframe.dtypes)
    
    # Variables numériques
    col_num = dataframe.select_dtypes(['int64', 'float64']).columns
    print("\nVariables numériques: ")
    print(col_num)
    
    # Variables catégorielles
    col_cat = dataframe.select_dtypes(['object']).columns
    print("\n Variables catégorielles: ")
    print(col_cat)
    
    # Calculer la proportion de valeurs manquantes pour chaque variable
    missing_proportions = dataframe.isnull().mean() * 100
    
    # Afficher les proportions dans un tableau lisible
    missing_summary = pd.DataFrame({
        "Variable": missing_proportions.index,
        "Proportion de valeurs manquantes (%)": missing_proportions.values
    }).sort_values(by="Proportion de valeurs manquantes (%)", ascending=False)
    
    print("\nProportion des valeurs manquantes:")
    print(missing_summary)
    
    # Afficher le nombre d'observations dans le DataFrame 
    print(f"Nombre d'observations : {dataframe.shape[0]}")



### 2-Fonction de verification de l'encodage numerique 
"""
Il est important de verifier notre notre encodage 
"""

def verification_normalisation(data_mod):
      for variable_norm in data_mod :
        data_mod_col=data_mod[variable_norm]
        moyenne=np.mean(data_mod_col)
        ecartType = np.std(data_mod_col)

        print("les statistiques pour la variable {}" .format(variable_norm))
        print("La moyenne est de : {} ".format(round(abs(moyenne), 2)))
        print("L'écart type est de : {} ".format(round(abs(ecartType), 2)))
        print(" ")
        print(" ")

### 3- Ecodage des varaiables catégorielles 

""" Prends en entrée les variables categoreille et numériques, 
encode les var catégorielle et retourne le dateset final constitué des var 
catégorielle et numérique sans retondnace
""" 
def encoder_onehot(dataframe, col_cat, col_num):
    
    # Initialiser l'encodeur OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    # Appliquer l'encodage One-Hot sur les colonnes catégorielles
    data_ohe = pd.DataFrame(encoder.fit_transform(dataframe[col_cat]).toarray())
    
    # Obtenir les noms des nouvelles caractéristiques
    feature_names = encoder.get_feature_names_out(input_features=col_cat)
    data_ohe.columns = feature_names
    
    # Supprimer les colonnes catégorielles d'origine et fusionner avec les variables numériques
    df_final = pd.concat([dataframe[col_num], data_ohe], axis=1)
    
    # Réinitialiser les index pour le DataFrame final
    df_final = df_final.reset_index(drop=True)
    print(df_final.columns)
    
    # Retourner le DataFrame final
    return df_final


## 4- Resumé de l'avaluation des modèles 

def evaluate_models_simple(models,X_train, X_test, y_train, y_test):
    # Initialisation des modèles

    # Évaluation des modèles
    result = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        result[name] = {"RMSE": rmse, "R2 Score": r2}

    # Résultats sous forme de DataFrame
    #print(pd.DataFrame(result))

    return result


