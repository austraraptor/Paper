import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('DATASET_REGISTRO_CONSUMO_COMBUSTIBLE_FLOTA_VEH_MPPAITA.csv')


df['KM_RECORRIDO_MENSUAL'] = df['KM_RECORRIDO_MENSUAL'].astype(str).str.replace(',', '').astype(float)
df['COSTO_UNI_COMBUSTIBLE'] = df['COSTO_UNI_COMBUSTIBLE'].astype(str).str.replace(',', '').astype(float)
df['COSTO_MENSUAL_CONSUMO'] = df['COSTO_MENSUAL_CONSUMO'].astype(str).str.replace(',', '').astype(float)


print("Valores nulos en el dataset:\n", df.isnull().sum())


df = df.dropna(subset=["KM_RECORRIDO_MENSUAL", "TIPO_COMBUSTIBLE", "NUM_DIAS_RECORRIDO", "COSTO_UNI_COMBUSTIBLE", "COSTO_MENSUAL_CONSUMO"])


X = df[["KM_RECORRIDO_MENSUAL", "TIPO_COMBUSTIBLE", "NUM_DIAS_RECORRIDO", "COSTO_UNI_COMBUSTIBLE"]]
y = df["COSTO_MENSUAL_CONSUMO"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['TIPO_COMBUSTIBLE']),
        ('num', 'passthrough', ['KM_RECORRIDO_MENSUAL', 'NUM_DIAS_RECORRIDO', 'COSTO_UNI_COMBUSTIBLE'])
    ])


modelo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# =========================
# 1. EVALUACIÓN DE MODELO CON VALIDACIÓN CRUZADA
# =========================
try:
    scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, 6), scores, color='skyblue')
    plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Promedio R² = {np.mean(scores):.2f}")
    plt.xlabel("Fold")
    plt.ylabel("R²")
    plt.title("Evaluación de modelos: R² por validación cruzada")
    plt.legend()
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print("Error en la validación cruzada:", e)

# =========================
# 2. PREDICCIÓN Y DIAGRAMA DE DISPERSIÓN
# =========================
try:
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Consumo real (soles)")
    plt.ylabel("Consumo predicho (soles)")
    plt.title("Dispersión: Consumo real vs. predicho (modelo Random Forest)")
    plt.grid(True)
    plt.show()
except ValueError as e:
    print("Error en el ajuste del modelo:", e)

# =========================
# 3. CUANTIFICACIÓN DE INCERTIDUMBRE
# =========================
try:
    
    n_iter = 100
    predicciones_bootstrap = []

    for i in range(n_iter):
        sample_indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        modelo.fit(X_sample, y_sample)
        pred = modelo.predict(X)
        predicciones_bootstrap.append(pred)

    pred_array = np.array(predicciones_bootstrap)
    y_pred_mean = pred_array.mean(axis=0)
    y_pred_std = pred_array.std(axis=0)

    
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred_std, bins=30, color='orange', edgecolor='black')
    plt.xlabel("Desviación estándar de la predicción (±)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de la incertidumbre en las predicciones (bootstrap)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print("Error en la cuantificación de incertidumbre:", e)

# =========================
# 4. VALIDACIÓN CON K-FOLD
# =========================
try:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    folds = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        modelo.fit(X_train, y_train)
        y_pred_k = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_k)
        mse_scores.append(mse)
        folds.append(f"Fold {i}")

  
    plt.figure(figsize=(6, 4))
    plt.plot(folds, mse_scores, marker='o', linestyle='-', color='green')
    plt.axhline(np.mean(mse_scores), color='red', linestyle='--', label=f"Promedio MSE = {np.mean(mse_scores):.2f}")
    plt.xlabel("Fold")
    plt.ylabel("Error cuadrático medio (MSE)")
    plt.title("Validación K-Fold: Error por fold")
    plt.legend()
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print("Error en la validación K-Fold:", e)


