# 2_train_model.py (versión con Hyperparameter Tuning para RandomForest)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV # <-- CAMBIO 1: Importar la herramienta de búsqueda
from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle

print("Iniciando la OPTIMIZACIÓN de RandomForest con Hyperparameter Tuning...")

try:
    df_modelo = pd.read_csv('datos_mensuales_para_modelo.csv')
except FileNotFoundError:
    print("Error: Asegúrate de ejecutar la versión mejorada de '1_feature_engineering.py'.")
    exit()

# 1. Dividir los datos
train = df_modelo[df_modelo['ano'] < 2011]
test = df_modelo[df_modelo['ano'] >= 2011]

target = 'cantidad_mensual'
features = [
    'precio', 'mes', 'trimestre', 'ano',
    'venta_mes_anterior', 'venta_mismo_mes_ano_anterior',
    'media_movil_3m', 'std_movil_3m'
]

X_train = train[features]
y_train = np.log1p(train[target])

X_test = test[features]
y_test = test[target]

print(f"Datos de entrenamiento: {X_train.shape[0]} filas.")

# --- CAMBIO 2: Definir el espacio de búsqueda de parámetros ---
# Estas son todas las combinaciones de "diales" que vamos a probar.
param_distributions = {
    'n_estimators': [100, 200, 300, 400],           # Número de árboles
    'max_depth': [10, 20, 30, None],               # Profundidad máxima de los árboles
    'min_samples_leaf': [2, 4, 6],                 # Mínimo de muestras en una hoja final
    'min_samples_split': [5, 10],                  # Mínimo de muestras para dividir un nodo
    'max_features': ['sqrt', 'log2', 1.0]          # Número de características a considerar en cada división
}

# 3. Configurar y ejecutar la búsqueda aleatoria
print("\nIniciando la búsqueda de los mejores hiperparámetros...")
rf = RandomForestRegressor(random_state=42)

# RandomizedSearchCV probará 50 combinaciones diferentes usando validación cruzada (cv=3)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,  # Probar 50 combinaciones de parámetros
    cv=3,       # Usar 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1   # Usar todos los núcleos de CPU disponibles
)

# Este paso tardará más, ya que está entrenando 150 modelos (50 iteraciones * 3 folds)
random_search.fit(X_train, y_train)

print("\nBúsqueda completada.")
print("Los mejores parámetros encontrados son:")
print(random_search.best_params_)

# --- CAMBIO 3: Usar el mejor modelo encontrado por la búsqueda ---
best_model = random_search.best_estimator_

# 4. Guardar el mejor modelo
nombre_archivo_modelo = 'modelo_prediccion_mensual.pkl'
with open(nombre_archivo_modelo, 'wb') as file:
    pickle.dump(best_model, file)
print(f"\n¡Mejor modelo RandomForest guardado exitosamente en '{nombre_archivo_modelo}'!")

# 5. Realizar y evaluar predicciones con el modelo optimizado
print("\nRealizando predicciones con el modelo optimizado...")
log_predictions = best_model.predict(X_test)
predictions = np.expm1(log_predictions)
predictions[predictions < 0] = 0

mae = mean_absolute_error(y_test, predictions)

print(f"\n--- Evaluación del Modelo RandomForest OPTIMIZADO ---")
print(f"El MAE final es de: {mae:.2f} unidades por mes.")