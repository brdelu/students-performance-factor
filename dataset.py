# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dataset_df = pd.read_csv("data_dropout.csv")
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
X = dataset_df.drop(columns="Target")
y = dataset_df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 )
X_train.shape

# %%
# créationd de la pipeline

# encoder la colonne Target car erreur sur le pipe.fit avec la lin reg ValueError: could not convert string to float: 'Graduate'

one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = one_hot_encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = one_hot_encoder.transform(y_test.values.reshape(-1, 1))

# %%
cat_features_selector = make_column_selector(dtype_include='object')
num_features_selector = make_column_selector(dtype_include=['float64', 'int64'])

# %%
preproc = make_column_transformer((StandardScaler(), num_features_selector), (OneHotEncoder(), cat_features_selector))
preproc

# %%
X_train_scaled = preproc.fit_transform(X_train)
X_train_scaled.shape

# %%
from sklearn.decomposition import PCA

# %%
preproc_with_pca = make_pipeline(preproc, PCA(n_components=4))
preproc_with_pca

# %%
X_train_scaled = preproc_with_pca.fit_transform(X_train)
X_train_scaled.shape

# %%
from sklearn.linear_model import LinearRegression

# %%
lin_pipe = make_pipeline(preproc_with_pca, LinearRegression())
lin_pipe

# %%
lin_pipe.fit(X_train, y_train_encoded)

# %%
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, classification_report

# %%
# prédictionx
predictions = lin_pipe.predict(X_test)

# %%
# Évaluation
mse = mean_squared_error(y_test_encoded, predictions)
print(f'Mean Squared Error (PCA + Regression): {mse}')


