import pandas as pd
import numpy as np
import torch
import os
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("🚀 Iniciando RE-ENTRENAMIENTO para corregir configuración...")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
scale_items = {
"CCFM": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 27, 40, 53, 66, 79, 92, 105, 118, 131],
"CCSS": [2, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 41, 54, 67, 80, 93, 106, 119, 132],
"CCNA": [3, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 42, 55, 68, 81, 94, 107, 120, 133],
"CCCO": [4, 17, 30, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 56, 69, 82, 95, 108, 121, 134],
"ARTE": [5, 18, 31, 44, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 70, 83, 96, 109, 122, 135],
"BURO": [6, 19, 32, 45, 58, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 84, 97, 110, 123, 136],
"CCEP": [7, 20, 33, 46, 59, 72, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 98, 111, 124, 137],
"HAA": [8, 21, 34, 47, 60, 73, 86, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 112, 125, 138],
"FINA": [9, 22, 35, 48, 53, 61, 74, 87, 100, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 126, 139],
"LING": [10, 23, 36, 49, 50, 62, 75, 88, 101, 114, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 140],
"JURI": [11, 24, 37, 50, 63, 63, 76, 89, 102, 115, 118, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]
    # Excluimos VERA y CONS para la predicción vocacional
}

# ==========================================
# 2. CARGAR DATOS
# ==========================================
archivo_excel = 'CASM83_limpio.xlsx'
if not os.path.exists(archivo_excel):
    print(f"❌ ERROR: Falta {archivo_excel}")
    exit()

df = pd.read_excel(archivo_excel)
df = df.fillna(0)

# ==========================================
# 3. DEFINIR TARGET (ETIQUETAS)
# ==========================================
df_scores = pd.DataFrame()
for escala, items in scale_items.items():
    cols = [f'Pregunta_{i}' for i in items]
    cols_existentes = [c for c in cols if c in df.columns]
    df_scores[escala] = df[cols_existentes].sum(axis=1)

df['Area_Dominante'] = df_scores.idxmax(axis=1)

# ==========================================
# 4. PREPARAR FEATURES (LO QUE VA AL MODELO)
# ==========================================
cols_preguntas = [f'Pregunta_{i}' for i in range(1, 144) if f'Pregunta_{i}' in df.columns]
# Esta es la lista que faltaba en tu archivo anterior:
features_cols = ['Genero', 'Grado'] + cols_preguntas

X = df[features_cols].values.astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(df['Area_Dominante'])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. ENTRENAR
# ==========================================
print("🧠 Entrenando modelo...")
clf = TabNetClassifier(verbose=1, optimizer_params=dict(lr=2e-2))
clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    patience=15, max_epochs=70,
    batch_size=128, virtual_batch_size=64
)

# ==========================================
# 6. GUARDAR CON LA LLAVE CORRECTA
# ==========================================
if not os.path.exists('modelos'):
    os.makedirs('modelos')

clf.save_model('modelos/tabnet_casm83_final')

# AQUÍ ESTÁ LA CORRECCIÓN: Guardamos 'features_cols' explícitamente
config = {
    'encoder': le,
    'features_cols': features_cols,  # <--- ESTO ES LO QUE FALTABA
    'scale_items': scale_items
}

with open('modelos/tabnet_config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("\n✅ CONFIGURACIÓN CORREGIDA Y GUARDADA.")
print("Ahora puedes ejecutar 'python app.py' sin errores.")