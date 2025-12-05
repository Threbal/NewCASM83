from flask import Flask, render_template, request
# Importamos la lista de preguntas del archivo que creamos en el Paso 1
from datos_preguntas import PREGUNTAS 

app = Flask(__name__)

# --- RUTA 1: Inicio ---
@app.route('/')
def inicio():
    return render_template('index.html')

# --- RUTA 2: Cuestionario (Antes se llamaba confirmar) ---
# Ahora, en lugar de solo mostrar datos, mostramos el cuestionario
@app.route('/cuestionario', methods=['POST'])
def cuestionario():
    # Recibimos los datos de la portada
    genero = request.form.get('genero')
    grado = request.form.get('grado')
    
    # Renderizamos la página del cuestionario enviándole las preguntas y los datos del usuario
    return render_template('cuestionario.html', 
                           preguntas=PREGUNTAS, 
                           genero=genero, 
                           grado=grado)

# --- RUTA 3: Resultados (Temporal) ---
@app.route('/resultados', methods=['POST'])
def resultados():
    # Aquí llegaremos cuando el usuario termine el test
    return "<h1>¡Felicidades! Has completado el test.</h1><p>Próximamente aquí verás tu resultado.</p>"

if __name__ == '__main__':
    app.run(debug=True)