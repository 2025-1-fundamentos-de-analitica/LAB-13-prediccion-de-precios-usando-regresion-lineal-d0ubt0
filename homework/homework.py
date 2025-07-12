#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def load_and_clean(path):
    df = pd.read_csv(path, compression="zip")
    df["Age"] = 2021 - df["Year"]
    # df["Age"] = df["Age"] ** 2

    df = df.drop(columns=["Year", "Car_Name"])

    df = df.dropna()
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]

    return X, y


def save_model_gzip(model, path="files/models/model.pkl.gz"):
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def save_regression_metrics(
    model, x_train, y_train, x_test, y_test, path="files/output/metrics.json"
):
    metrics = []
    model = model.best_estimator_

    for X, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(X)
        entry = {
            "type": "metrics",
            "dataset": name,
            "r2": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred),
            "mad": mean_absolute_error(y, y_pred),
        }
        metrics.append(entry)

    with open(path, "w") as f:
        for entry in metrics:
            json.dump(entry, f)
            f.write("\n")


x_train, y_train = load_and_clean("files/input/train_data.csv.zip")
x_test, y_test = load_and_clean("files/input/test_data.csv.zip")

print(x_test.columns)
print(x_train.head(), y_train.head())

categorical_features = ["Fuel_Type", "Selling_type", "Transmission", "Owner"]

numerical_features = [col for col in x_train.columns if col not in categorical_features]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", MinMaxScaler(), numerical_features),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(f_regression)),
        ("regressor", LinearRegression()),
    ]
)


param_grid = {"selector__k": list(range(2, 25))}

cv = KFold(n_splits=10, shuffle=True)

scorer = make_scorer(mean_absolute_error, greater_is_better=False)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=4
)

grid_search.fit(x_train, y_train)


print(grid_search.best_params_)
save_model_gzip(grid_search)

save_regression_metrics(grid_search, x_train, y_train, x_test, y_test)
