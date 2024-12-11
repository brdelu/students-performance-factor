# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dataset_df = pd.read_csv("data_dropout_ml_good.csv")
dataset_df.info()

# %%
# Calculer les quartiles pour les variables numériques
numerical_columns = dataset_df.columns[0:34].tolist()

# Obtenir les quartiles pour les variables numériques
quartiles = dataset_df[numerical_columns].quantile([0.25, 0.5, 0.75])

# Afficher les quartiles
print("Quartiles pour chaque variable:\n", quartiles)

# %%
# Corrélations entre les variables
correlation_matrix = dataset_df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matrice de corrélation des variables numériques")
plt.show()

# %%
dataset_df.head()

# %%
print(dataset_df['Gender'].nunique())
print(dataset_df['Gender'].unique())

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline 
from sklearn.compose import make_column_selector, make_column_transformer

# %%
# Liste des colonnes à convertir en numériques
numeric_columns = [
    'Previous qualification /10', 'CU 1st sem (grade, number)',  
    'Unemployment rate', 'Inflation rate', 'GDP', 'CU 2nd sem (grade, number)', 'Admission grade'
]

# Convertir les colonnes numériques en types numériques en remplaçant les virgules par des points
for col in numeric_columns:
    # Convertir d'abord en chaîne de caractères, puis remplacer les virgules par des points si nécessaire
    dataset_df[col] = dataset_df[col].astype(str).str.replace(',', '.')
    
    # Essayer de convertir les valeurs en float, en forçant la gestion des erreurs
    dataset_df[col] = pd.to_numeric(dataset_df[col], errors='coerce')
    
# Afficher les informations après conversion pour vérifier
dataset_df.info()

# Vérifier les premières lignes pour s'assurer que la conversion s'est bien déroulée
dataset_df.head()

# %%
dataset_df = dataset_df.dropna()
dataset_df.shape

# %%
X = dataset_df.drop(columns="Target")
y = dataset_df["Target"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 )
X_train.shape

# %% [markdown]
# perte de 4424 - 3539 = 885 lignes

# %%
# 4. Vérifier la présence de NaN dans X_train et X_test
print("NaN dans X_train avant transformation :")
print(X_train.isna().sum())
print("NaN dans X_test avant transformation :")
print(X_test.isna().sum())

# %%
# création de la pipeline

# encoder la colonne Target car erreur sur le pipe.fit avec application lin_reg "ValueError: could not convert string to float: 'Graduate'"

one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = one_hot_encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = one_hot_encoder.transform(y_test.values.reshape(-1, 1))

# %%
cat_features_selector = make_column_selector(dtype_include='object')
num_features_selector = make_column_selector(dtype_include=['float64', 'int64'])

# %%
# Sélectionner les colonnes catégorielles avec make_column_selector
cat_columns = cat_features_selector(X_train)  # On passe X_train pour sélectionner les colonnes dans le DataFrame

# Vérifier les catégories uniques pour chaque colonne catégorielle
for col in cat_columns:
    print(f"Nombre de catégories pour {col}: {X_train[col].nunique()}")
    print(f"Valeurs uniques pour {col}: {X_train[col].unique()}")

# %% [markdown]
# from sklearn.impute import SimpleImputer
# preproc = make_column_transformer(
#     (SimpleImputer(strategy='mean'), num_features_selector),
#     (StandardScaler(), num_features_selector), 
#     (OneHotEncoder(handle_unknown='ignore'), cat_features_selector)
#     )
# preproc

# %%
from sklearn.impute import SimpleImputer
preproc = make_column_transformer(
    (StandardScaler(), num_features_selector), 
    (OneHotEncoder(handle_unknown='ignore'), cat_features_selector)
    )
preproc

# %%
X_train_scaled = preproc.fit_transform(X_train)
X_train_scaled.shape

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# créer pipeline pour random forest
pipeline_rf = make_pipeline(preproc, RandomForestClassifier(n_estimators=100, random_state=0))
pipeline_rf.fit(X_train, y_train_encoded)

# %% [markdown]
# À partir des informations fournies, il semble que y_train contient des valeurs de type object (chaînes de caractères), ce qui indique qu'il s'agit d'une classification multiclass avec trois classes possibles : Graduate, Enrolled, et Dropout.

# %%
from sklearn.linear_model import LinearRegression

# créer pipeline pour linear regression
pipeline_lr = make_pipeline(
    preproc,  # Appliquer le préprocessing
    LinearRegression()  # Appliquer le modèle Régression Linéaire
)

# 7. Entraîner le modèle Régression Linéaire
pipeline_lr.fit(X_train, y_train_encoded)

# %%
# créer une pipeline parllèle avec voting classifier pour combiner les 2

# %%
# Calculer la Permutation Feature Importance pour RandomForest
perm_importance = permutation_importance(pipeline_rf, X_test, y_test_encoded, n_repeats=10, random_state=0)

# 8. Afficher les résultats de l'importance des caractéristiques
print("Permutation Feature Importance :")
# Extraire l'importance des caractéristiques à partir de la pipeline
feature_names = X.columns  # Récupérer les noms des caractéristiques d'origine
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]:<30} {perm_importance.importances_mean[i]:.3f} +/- {perm_importance.importances_std[i]:.3f}")

# %%
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# comparaisons des 2 modèles
# random forest
y_pred_rf = pipeline_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test_encoded, y_pred_rf)

# régression linéaire
y_pred_lr = pipeline_lr.predict(X_test)
mse_lr = mean_squared_error(y_test_encoded, y_pred_lr)
r2_lr = r2_score(y_test_encoded, y_pred_lr)

# Affichage des résultats de comparaison
print("\nComparaison des performances:")
print(f"Accuracy du Random Forest : {accuracy_rf:.3f}")
print(f"MSE de la Régression Linéaire : {mse_lr:.3f}")
print(f"R2 de la Régression Linéaire : {r2_lr:.3f}")

# %%
# mise en place d'une gridSearch pour fin tune les parmaètres de la random forest

from sklearn.model_selection import GridSearchCV

param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [10, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline_rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train_encoded)

# Afficher les meilleurs paramètres
print("Meilleurs paramètres trouvés : ", grid_search.best_params_)

# Afficher la meilleure performance
print("Meilleure performance : ", grid_search.best_score_)


