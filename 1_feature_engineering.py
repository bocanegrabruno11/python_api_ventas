# 1_feature_engineering.py (versión MEJORADA)

import pandas as pd

print("Iniciando la ingeniería de características mensuales AVANZADA...")

try:
    df_agregado = pd.read_csv('datos_mensuales_limpios.csv')
except FileNotFoundError:
    print("Error: Asegúrate de ejecutar primero el script de limpieza mensual.")
    exit()

df_agregado['fecha'] = pd.to_datetime(df_agregado['fecha'])
df_modelo = df_agregado.copy()
df_modelo['precio'].fillna(method='ffill', inplace=True)

# 1. Características de tiempo (sin cambios)
print("Creando características de tiempo...")
df_modelo['mes'] = df_modelo['fecha'].dt.month
df_modelo['trimestre'] = df_modelo['fecha'].dt.quarter
df_modelo['ano'] = df_modelo['fecha'].dt.year

# 2. Características de desfase (Lag Features) mejoradas
print("Creando características de desfase (lags) mensuales...")
df_modelo['venta_mes_anterior'] = df_modelo.groupby('codigo_stock')['cantidad_mensual'].shift(1)
# <-- NUEVA CARACTERÍSTICA: Estacionalidad anual
df_modelo['venta_mismo_mes_ano_anterior'] = df_modelo.groupby('codigo_stock')['cantidad_mensual'].shift(12)

# 3. Características de ventana móvil (Rolling Window) mejoradas
print("Creando características de ventana móvil mensuales...")
df_modelo['media_movil_3m'] = df_modelo.groupby('codigo_stock')['cantidad_mensual'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)
# <-- NUEVA CARACTERÍSTICA: Volatilidad de ventas
df_modelo['std_movil_3m'] = df_modelo.groupby('codigo_stock')['cantidad_mensual'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
)


# Rellenamos los valores nulos (NaN) generados al principio con 0
df_modelo.fillna(0, inplace=True)

# Guardar el dataset final
nombre_archivo_final = 'datos_mensuales_para_modelo.csv'
df_modelo.to_csv(nombre_archivo_final, index=False)

print("\n--- Muestra del dataset final con características mejoradas ---")
print(df_modelo.head())
print(f"\n¡Proceso completado! Se ha guardado el dataset para el modelo en '{nombre_archivo_final}'")