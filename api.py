# api.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import numpy as np
import traceback
from dotenv import load_dotenv 
import openai 

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
COLUMNAS_MODELO = [
    'precio',
    'mes', 
    'trimestre', 
    'ano',
    'venta_mes_anterior',
    'venta_mismo_mes_ano_anterior', # <-- Faltaba esta
    'media_movil_3m',
    'std_movil_3m'                  # <-- Faltaba esta
]
app = Flask(__name__)

model = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Asegúrate de que este sea el modelo entrenado con RandomizedSearchCV
    modelo_path = os.path.join(script_dir, 'modelo_prediccion_mensual.pkl')
    with open(modelo_path, 'rb') as file:
        model = pickle.load(file)
    print("--- Modelo de predicción MENSUAL OPTIMIZADO cargado exitosamente ---")
except Exception as e:
    print(f"!!! ERROR CRÍTICO AL CARGAR EL MODELO: {str(e)} !!!")


def predecir(lista_productos, meses_a_predecir=1):
    if model is None: return {"error": "El modelo no está cargado."}

    resultados = []
    for datos_producto in lista_productos:
        try:
            cantidad_total_predicha = 0
            detalle_mensual = []
            # --- CAMBIO 2: Recoger TODAS las características del payload de Laravel ---
            features_iterativas = {
                'precio': datos_producto['precio_actual'],
                'venta_mes_anterior': datos_producto['venta_mes_anterior'],
                'venta_mismo_mes_ano_anterior': datos_producto['venta_mismo_mes_ano_anterior'],
                'media_movil_3m': datos_producto['media_movil_3m'],
                'std_movil_3m': datos_producto['std_movil_3m'],
            }
            
            fecha_actual = pd.to_datetime(datos_producto['fecha_a_predecir']).to_period('M').to_timestamp()

            for _ in range(meses_a_predecir):
                # 1. Actualizar características de tiempo
                features_iterativas['mes'] = fecha_actual.month
                features_iterativas['trimestre'] = fecha_actual.quarter
                features_iterativas['ano'] = fecha_actual.year
                
                # 2. Predecir
                df_features = pd.DataFrame([features_iterativas])
                df_features = df_features[COLUMNAS_MODELO] # Reordenar
                
                log_prediccion = model.predict(df_features)
                
                # --- CAMBIO 3: Revertir la transformación logarítmica ---
                cantidad_predicha_mes = np.expm1(log_prediccion[0])
                cantidad_predicha_mes = round(float(cantidad_predicha_mes))

                if cantidad_predicha_mes < 0: cantidad_predicha_mes = 0
                
                cantidad_total_predicha += cantidad_predicha_mes
                detalle_mensual.append({
                    'fecha': fecha_actual.strftime('%Y-%m-%d'), # Guardamos la fecha del mes
                    'cantidad': cantidad_predicha_mes
                })
                
                # 4. Preparar la siguiente iteración
                features_iterativas['venta_mes_anterior'] = cantidad_predicha_mes
                
                fecha_actual = (fecha_actual.to_period('M') + 1).to_timestamp()

            resultados.append({
                'codigo_stock': datos_producto['codigo_stock'],
                'cantidad_total_predicha': cantidad_total_predicha,
                'detalle_mensual': detalle_mensual 
            })
        except Exception as e:
            resultados.append({
                'codigo_stock': datos_producto.get('codigo_stock', 'desconocido'), 
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
    return {"predicciones": resultados}



# --- 4. NUEVA LÓGICA PARA SUGERENCIAS CON OPENAI --- 🤖
def obtener_sugerencia_ia(nombre, categoria, stock, ventas_mes, rotacion,mes):
    try:
        # Prompt mejorado con más contexto para mejores respuestas
        prompt = f"""
        Actúa como un Analista Estratégico de Inventarios para una empresa comercial en Trujillo, Perú. Tu objetivo es maximizar la rentabilidad y la eficiencia del inventario.

        Proporciona una recomendación de negocio profesional, concisa y accionable en español para el siguiente producto, considerando que estamos en el mes {mes} del año.
        Al tratarse de ventas, ten en cuenta el tipo de producto (Categoría: {categoria}) y considera las estaciones del año y las temporadas de productos existentes en el país.

        **Datos del Producto para el Mes de Evaluación (Mes {mes}):**
        - Nombre: {nombre}
        - Categoría: {categoria}
        - Stock Actual (al final del mes): {stock} unidades
        - Ventas Totales del Mes: {ventas_mes} unidades
        - Índice de Rotación del Mes: {rotacion}%  (Calculado como: Ventas / Stock Promedio * 100)

        **Tu Tarea: Analizar los Datos**
        Tu objetivo es analizar la *relación* entre las ventas, el stock actual y el índice de rotación.
        
        **Contexto Crítico para tu Análisis:**
        1.  **No uses reglas simplistas.** El índice de rotación es una guía, pero los números brutos son la verdad.
        2.  **Caso de Alerta (Exceso de Stock):** Si las ventas son una fracción diminuta del stock (ej. Ventas=15, Stock=1500), esto es un **problema grave de exceso de stock**, sin importar la categoría. Una rotación tan baja (ej. 1%) es inaceptable.
        3.  **Caso Saludable:** Si las ventas son una porción significativa del stock (ej. Ventas=140, Stock=300), la situación es mucho más saludable, aunque la rotación (ej. ~40%) sea menor a 100%. Tu análisis debe reflejar esta diferencia.
        4.  **Prioriza el Sentido Común:** Utiliza la Categoría y el Mes para *matizar* tu recomendación, no para ignorar un problema claro de stock (como en el Caso de Alerta).

        Basado en un análisis crítico de estos datos, dame una sugerencia estratégica directa. Al final, me debes devolver la respuesta con este formato:
        Diagnóstico: [Tu análisis de la situación: exceso de stock, stock saludable, riesgo de quiebre, etc.]
        Acción Recomendada: [Tu sugerencia accionable: liquidar stock, reponer, pausar compras, etc.]
        Justificación: [Explica brevemente por qué das esa recomendación, basándote en la relación Ventas vs. Stock.]
        Sugerencia:
        """

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un analista de inventarios experto que basa sus respuestas en un análisis numérico crítico."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=250, # Aumentamos un poco por si la respuesta es más elaborada
        )
        
        sugerencia = completion.choices[0].message.content.strip()
        return {"sugerencia": sugerencia}
    except Exception as e:
        return {"error": f"Error al contactar la API de OpenAI: {str(e)}"}


# --- 5. RUTAS DE LA API (ENDPOINTS) ---

# Ruta para la predicción de demanda (tu código existente, sin cambios)
@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        input_data = request.get_json()
        if not input_data or 'productos' not in input_data:
            return jsonify({"error": "Payload inválido."}), 400
        
        meses_para_predecir = int(input_data.get('meses_a_predecir', 1))

        resultado = predecir(input_data['productos'], meses_para_predecir)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": "Error interno.", "details": str(e)}), 500

# Nueva ruta para las sugerencias de OpenAI
@app.route('/suggest', methods=['POST'])
def handle_suggestion():
    try:
        input_data = request.get_json()
        
        # Validamos que todos los nuevos campos estén presentes
        required_keys = ["nombre", "categoria", "stock", "ventas_mes", "rotacion"]
        if not all(k in input_data for k in required_keys):
            return jsonify({"error": f"Faltan datos. Se requieren: {', '.join(required_keys)}"}), 400
        
        resultado = obtener_sugerencia_ia(
            input_data["nombre"], 
            input_data["categoria"],
            input_data["stock"],
            input_data["ventas_mes"],
            input_data["rotacion"],
            input_data["mes"],


        )

        if "error" in resultado:
            return jsonify(resultado), 500

        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": "Error interno en el endpoint de sugerencias.", "details": str(e)}), 500


# --- INICIAR EL SERVIDOR (CONFIGURACIÓN RAILWAY) ---
if __name__ == '__main__':
    # Obtenemos el puerto de Railway, por defecto 5000 si es local
    port = int(os.environ.get("PORT", 5000))
    # '0.0.0.0' es OBLIGATORIO para Docker/Railway
    app.run(host='0.0.0.0', port=port)