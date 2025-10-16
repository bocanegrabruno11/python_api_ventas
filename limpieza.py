# limpieza.py (versión mensual)

import pandas as pd

# Paso 1: Cargar el dataset
ruta_archivo = r'C:\Users\bocan\OneDrive\Escritorio\Tesis Archivos Software\Modelo IA\dirty_online_retail.csv'
try:
    df = pd.read_csv(ruta_archivo, encoding='windows-1252', dtype={'codigo_stock': str})
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en '{ruta_archivo}'. Verifica la ruta.")
    exit()

print(f"Dataset cargado con {df.shape[0]} filas.")

# --- FASE 3: PREPARACIÓN DE DATOS (CRISP-DM) ---
df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
df.dropna(subset=['codigo_stock'], inplace=True)
df = df[df['cantidad'] > 0]
df = df[df['precio'] > 0]

print(f"Dataset después de la limpieza tiene {df.shape[0]} filas.")

df_esencial = df[['fecha', 'codigo_stock', 'cantidad', 'precio']]
df_esencial = df_esencial.sort_values(by='fecha')
df_esencial.reset_index(drop=True, inplace=True)

# --- CAMBIO CLAVE: Agregación de datos por MES ---
print("\nAgregando el dataset a nivel mensual...")
# Usamos 'MS' para agrupar por el inicio de cada mes
df_agregado = df_esencial.groupby('codigo_stock').resample('MS', on='fecha').agg({
    'cantidad': 'sum',
    'precio': 'mean'
}).reset_index()

# Renombramos la columna para que sea más claro
df_agregado.rename(columns={'cantidad': 'cantidad_mensual'}, inplace=True)

print("\n--- Muestra del dataset agregado mensualmente ---")
print(df_agregado.head())

# Guardamos el nuevo archivo con un nombre diferente
nombre_archivo_limpio = 'datos_mensuales_limpios.csv'
df_agregado.to_csv(nombre_archivo_limpio, index=False)

print(f"\n¡Proceso completado! Se ha guardado el dataset mensual en '{nombre_archivo_limpio}'")