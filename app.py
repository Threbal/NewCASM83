from flask import Flask, render_template, request
import numpy as np
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from datos_preguntas import PREGUNTAS 

app = Flask(__name__)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
modelo_listo = False
modelo_ia = None
le = None
features_cols = []
scale_items = {}

# Definimos los nombres bonitos
nombres_areas = {
    'CCFM': 'Ciencias Físico-Matemáticas', 'CCSS': 'Ciencias Sociales',
    'CCNA': 'Ciencias Naturales', 'CCCO': 'Ciencias de la Comunicación',
    'ARTE': 'Artes', 'BURO': 'Burocracia', 'CCEP': 'Ciencias Políticas',
    'HAA':  'Institutos Armados', 'FINA': 'Finanzas', 'LING': 'Lingüística',
    'JURI': 'Jurídica', 'VERA': 'Veracidad', 'CONS': 'Consistencia'
}

# Cargar el Modelo Entrenado
print("⏳ Cargando Cerebro IA...")
try:
    with open('modelos/tabnet_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    le = config['encoder']
    features_cols = config['features_cols']  # Las columnas exactas del entrenamiento
    scale_items = config['scale_items']      # Necesario para sumar puntos
    
    modelo_ia = TabNetClassifier()
    modelo_ia.load_model('modelos/tabnet_casm83_final.zip')
    
    modelo_listo = True
    print("✅ IA TabNet cargada y lista.")
except Exception as e:
    print(f"⚠️ Error cargando IA: {e}")
    print("Asegúrate de ejecutar 'crear_modelo_tabnet.py' primero.")
    modelo_listo = False

# ==========================================
# 2. RUTAS
# ==========================================
@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/cuestionario', methods=['POST'])
def cuestionario():
    genero = request.form.get('genero')
    grado = request.form.get('grado')
    return render_template('cuestionario.html', preguntas=PREGUNTAS, genero=genero, grado=grado)

@app.route('/resultados', methods=['POST'])
def resultados():
    if not modelo_listo:
        return "<h3>Error: La IA no está lista. Ejecuta el script de entrenamiento.</h3>"

    datos = request.form
    
    # --- A. PREPARAR DATOS ---
    input_data = {}
    try:
        input_data['Genero'] = int(datos.get('genero', 0))
        input_data['Grado'] = int(datos.get('grado', 0))
    except:
        input_data['Genero'] = 0
        input_data['Grado'] = 0
    
    # Guardamos respuestas crudas (0-3) para calcular puntos después
    respuestas_crudas = {}
    for i in range(1, 144):
        val = datos.get(f'Pregunta_{i}', 0)
        valor_int = int(val) if val else 0
        input_data[f'Pregunta_{i}'] = valor_int
        respuestas_crudas[f'Pregunta_{i}'] = valor_int

    # --- B. CONSULTAR A LA IA (¿QUIÉN GANA?) ---
    # Creamos el vector en el orden exacto que aprendió la IA
    vector = []
    for col in features_cols:
        vector.append(input_data.get(col, 0))
        
    X_input = np.array([vector], dtype=np.float32)
    
    # Obtenemos las probabilidades (La opinión de la IA)
    probs = modelo_ia.predict_proba(X_input)[0]
    
    # Creamos un ranking basado en la certeza de la IA
    # Lista de tuplas: [('CCFM', 0.85), ('ARTE', 0.10)...]
    ranking_ia = []
    for idx, prob in enumerate(probs):
        codigo_area = le.inverse_transform([idx])[0]
        ranking_ia.append((codigo_area, prob))
    
    # Ordenamos: La IA decide el orden
    ranking_ia.sort(key=lambda x: x[1], reverse=True)
    
    # Nos quedamos con los 3 códigos ganadores
    top_3_codigos = [item[0] for item in ranking_ia[:3]]
    
    # --- C. CALCULAR PUNTOS (SOLO PARA LOS GANADORES) ---
    # Ahora que sabemos quién ganó, calculamos sus puntos reales para mostrarlos
    top_3_final = []
    
    for codigo in top_3_codigos:
        nombre_largo = nombres_areas.get(codigo, codigo)
        
        # Suma manual de puntos para este área
        puntos_reales = 0
        preguntas_area = scale_items.get(codigo, [])
        for p in preguntas_area:
            puntos_reales += respuestas_crudas.get(f'Pregunta_{p}', 0)
            
        top_3_final.append((nombre_largo, puntos_reales))
    
    # El ganador (Puesto 1)
    ganador_nombre = top_3_final[0][0]
    ganador_puntaje = top_3_final[0][1]

    # --- D. ENVIAR A PANTALLA ---
    return render_template('resultados.html', 
                           ganador_nombre=ganador_nombre,
                           ganador_puntaje=ganador_puntaje,
                           top_3=top_3_final)

if __name__ == '__main__':
    app.run(debug=True)