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


df = df.dropna(subset=["KM_RECORRIDO_MENSUAL", "TIPO_COMBUSTIBLE", "NUM_DIAS_RECORRIDO", "COSTO_UNI_COMBUSTIBLE", "COSTO_MENSUAL_CONSUMO"])


X = df[["KM_RECORRIDO_MENSUAL", "TIPO_COMBUSTIBLE", "NUM_DIAS_RECORRIDO", "COSTO_UNI_COMBUSTIBLE"]]
y = df["COSTO_MENSUAL_CONSUMO"]


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['TIPO_COMBUSTIBLE']),
    ('num', 'passthrough', ['KM_RECORRIDO_MENSUAL', 'NUM_DIAS_RECORRIDO', 'COSTO_UNI_COMBUSTIBLE'])
])


modelo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ========== 1. VALIDACIÓN CRUZADA R² ==========
try:
    scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')

    print("\n=== RESULTADOS R² VALIDACIÓN CRUZADA (5 FOLDS) ===")
    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: R² = {score:.4f}")
    print(f"R² promedio: {np.mean(scores):.4f}")

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

# ========== 2. REAL VS. PREDICHO ==========
try:
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    print("\n=== RESULTADOS REAL VS. PREDICHO ===")
    errores = y - y_pred
    print(f"Error absoluto medio (MAE): {np.mean(np.abs(errores)):.2f}")
    print(f"Error cuadrático medio (MSE): {mean_squared_error(y, y_pred):,.2f}")
    print(f"Coeficiente de determinación (R²): {r2_score(y, y_pred):.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Consumo real (soles)")
    plt.ylabel("Consumo predicho (soles)")
    plt.title("Dispersión: Consumo real vs. predicho (modelo Random Forest)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print("Error en el ajuste del modelo:", e)

# ========== 3. INCERTIDUMBRE BOOTSTRAP ==========
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
    y_pred_std = pred_array.std(axis=0)

    print("\n=== RESULTADOS INCERTIDUMBRE (BOOTSTRAP) ===")
    print(f"Desviación estándar media: ±{np.mean(y_pred_std):.2f}")
    print(f"Desviación estándar mínima: ±{np.min(y_pred_std):.2f}")
    print(f"Desviación estándar máxima: ±{np.max(y_pred_std):.2f}")
    print(f"Percentil 25 (P25): ±{np.percentile(y_pred_std, 25):.2f}")
    print(f"Percentil 75 (P75): ±{np.percentile(y_pred_std, 75):.2f}")

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

# ========== 4. VALIDACIÓN K-FOLD MSE ==========
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

    mse_promedio = np.mean(mse_scores)

    print("\n=== RESULTADOS MSE VALIDACIÓN K-FOLD ===")
    for f, mse in zip(folds, mse_scores):
        print(f"{f}: MSE = {mse:,.2f} Soles²")
    print(f"MSE promedio (5-Fold): {mse_promedio:,.2f} Soles²")

    plt.figure(figsize=(6, 4))
    plt.plot(folds, mse_scores, marker='o', linestyle='-', color='green')
    plt.axhline(mse_promedio, color='red', linestyle='--', label=f"Promedio MSE = {mse_promedio:,.2f}")
    plt.xlabel("Fold")
    plt.ylabel("Error cuadrático medio (MSE)")
    plt.title(f"Validación K-Fold: MSE por fold\n(MSE promedio = {mse_promedio:,.2f} Soles²)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
except ValueError as e:
    print("Error en la validación K-Fold:", e)


