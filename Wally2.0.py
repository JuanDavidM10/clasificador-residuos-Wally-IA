"""
╔══════════════════════════════════════════════════════════════╗
║          Wally AI - CLASIFICADOR DE RESIDUOS             ║
║                    Versión Streamlit                         ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from datetime import datetime
from collections import Counter
import os
from PIL import Image

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Wally AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
IMG_SIZE = (256, 256)

# Parámetros HOG
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)
HOG_BLOCK_NORM = "L2-Hys"

# Colores y metadatos
CLASS_INFO = {
    "cardboard": {
        "nombre": "CARTÓN",
        "emoji": "📦",
        "color": "#8B4513",
        "descripcion": "Reciclable",
        "contenedor": "Contenedor Azul"
    },
    "glass": {
        "nombre": "VIDRIO",
        "emoji": "🍾",
        "color": "#4FC3F7",
        "descripcion": "Reciclable",
        "contenedor": "Contenedor Verde"
    },
    "metal": {
        "nombre": "METAL",
        "emoji": "🔩",
        "color": "#9E9E9E",
        "descripcion": "Reciclable",
        "contenedor": "Contenedor Amarillo"
    },
    "paper": {
        "nombre": "PAPEL",
        "emoji": "📄",
        "color": "#FFD700",
        "descripcion": "Reciclable",
        "contenedor": "Contenedor Azul"
    },
    "plastic": {
        "nombre": "PLÁSTICO",
        "emoji": "🧴",
        "color": "#E91E63",
        "descripcion": "Reciclable con precaución",
        "contenedor": "Contenedor Amarillo"
    }
}

# ============================================================
# CSS PERSONALIZADO
# ============================================================

st.markdown("""
<style>
    /* Tema oscuro personalizado */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Título principal */
    .main-title {
        text-align: center;
        color: #10B981;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Tarjetas de resultado */
    .result-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 2px solid #10B981;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
        margin: 1rem 0;
    }
    
    .result-emoji {
        font-size: 5rem;
        margin: 1rem 0;
    }
    
    .result-class {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .result-confidence {
        font-size: 3rem;
        font-weight: 800;
        color: #10B981;
        margin: 0.5rem 0;
    }
    
    .result-info {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Métricas */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Botones */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1E293B;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE CARGA DE MODELO
# ============================================================

@st.cache_resource
def cargar_modelos():
    """Carga los modelos entrenados (solo una vez)"""
    try:
        modelo = joblib.load("modelo_final.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        return modelo, scaler, pca, None
    except Exception as e:
        return None, None, None, str(e)

# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def extraer_hog(img):
    """Extrae características HOG de una imagen"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True
    )
    return features

def clasificar_imagen(img, modelo, scaler, pca):
    """Clasifica una imagen y retorna predicción y probabilidades"""
    # Redimensionar
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Extraer HOG
    features = extraer_hog(img_resized)
    features = features.reshape(1, -1)
    
    # Normalizar y PCA
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    # Predecir
    prediction = modelo.predict(features_pca)[0]
    probabilities = modelo.predict_proba(features_pca)[0]
    
    return prediction, probabilities

# ============================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================

if 'historial' not in st.session_state:
    st.session_state.historial = []
    
if 'contador' not in st.session_state:
    st.session_state.contador = Counter()

# ============================================================
# HEADER
# ============================================================

st.markdown('<h1 class="main-title">♻️ Wally AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Wally AI - Clasificación Inteligente de Residuos en Tiempo Real</p>', unsafe_allow_html=True)

# ============================================================
# CARGAR MODELOS
# ============================================================

modelo, scaler, pca, error = cargar_modelos()

if error:
    st.error(f"""
    ❌ **Error al cargar los modelos**
    
    {error}
    
    Asegúrate de tener estos archivos en el mismo directorio:
    - `modelo_final.pkl`
    - `scaler.pkl`
    - `pca.pkl`
    """)
    st.stop()

# ============================================================
# SIDEBAR - INFORMACIÓN Y ESTADÍSTICAS
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Panel de Control")
    
    # Estado del modelo
    st.success("🟢 Modelo cargado correctamente")
    
    st.markdown("---")
    
    # Estadísticas
    st.markdown("### 📈 Estadísticas de Sesión")
    
    total = sum(st.session_state.contador.values())
    st.metric("Total de Clasificaciones", total)
    
    st.markdown("#### Por Clase:")
    for clase in CLASSES:
        info = CLASS_INFO[clase]
        count = st.session_state.contador.get(clase, 0)
        st.metric(
            f"{info['emoji']} {info['nombre']}",
            count
        )
    
    st.markdown("---")
    
    # Botón limpiar
    if st.button("🗑️ Limpiar Estadísticas", use_container_width=True):
        st.session_state.historial = []
        st.session_state.contador = Counter()
        st.rerun()
    
    st.markdown("---")
    
    # Información
    st.markdown("### ℹ️ Información")
    st.markdown("""
    **Clases detectables:**
    - 📦 Cartón
    - 🍾 Vidrio  
    - 🔩 Metal
    - 📄 Papel
    - 🧴 Plástico
    
    **Tecnología:**
    - Modelo: SVM + HOG
    - Precisión: ~87%
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado con ❤️ para IA")

# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

# Crear dos columnas
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📹 Captura de Imagen")
    
    # Tabs para diferentes opciones de entrada
    tab1, tab2, tab3 = st.tabs(["📸 Cámara Web", "📁 Subir Imagen", "🎥 Cámara en Tiempo Real"])
    
    with tab1:
        st.info("👇 Click en 'Take Photo' para capturar desde tu cámara")
        img_camera = st.camera_input("Captura una imagen")
        
        if img_camera:
            # Convertir a OpenCV format
            image = Image.open(img_camera)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Mostrar imagen capturada
            st.image(image, caption="Imagen Capturada", use_column_width=True)
            
            # Botón clasificar
            if st.button("🔍 Clasificar Imagen", key="classify_camera", use_container_width=True):
                with st.spinner("Clasificando..."):
                    prediction, probabilities = clasificar_imagen(img_bgr, modelo, scaler, pca)
                    
                    # Guardar en session state
                    st.session_state.ultima_prediccion = {
                        'clase': CLASSES[prediction],
                        'confianza': probabilities[prediction],
                        'probabilidades': probabilities,
                        'timestamp': datetime.now()
                    }
                    
                    # Actualizar estadísticas
                    st.session_state.contador[CLASSES[prediction]] += 1
                    st.session_state.historial.append(st.session_state.ultima_prediccion)
                    
                    st.rerun()
    
    with tab2:
        st.info("📂 Sube una imagen desde tu computadora")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            # Leer imagen
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Mostrar imagen
            st.image(image, caption="Imagen Cargada", use_column_width=True)
            
            # Botón clasificar
            if st.button("🔍 Clasificar Imagen", key="classify_upload", use_container_width=True):
                with st.spinner("Clasificando..."):
                    prediction, probabilities = clasificar_imagen(img_bgr, modelo, scaler, pca)
                    
                    st.session_state.ultima_prediccion = {
                        'clase': CLASSES[prediction],
                        'confianza': probabilities[prediction],
                        'probabilidades': probabilities,
                        'timestamp': datetime.now()
                    }
                    
                    st.session_state.contador[CLASSES[prediction]] += 1
                    st.session_state.historial.append(st.session_state.ultima_prediccion)
                    
                    st.rerun()
    
    with tab3:
        st.warning("⚠️ La cámara en tiempo real continuo requiere ejecución local.")
        st.markdown("""
        Para usar cámara en tiempo real:
        1. Descarga el código
        2. Ejecuta localmente: `streamlit run app_streamlit.py`
        3. Usa la pestaña "📸 Cámara Web" para capturas individuales
        """)

with col2:
    st.markdown("### 🎯 Resultado de Clasificación")
    
    # Mostrar último resultado si existe
    if 'ultima_prediccion' in st.session_state:
        pred = st.session_state.ultima_prediccion
        clase = pred['clase']
        confianza = pred['confianza']
        info = CLASS_INFO[clase]
        
        # Tarjeta de resultado
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{info['emoji']}</div>
            <div class="result-class" style="color: {info['color']}">{info['nombre']}</div>
            <div class="result-confidence">{confianza*100:.1f}%</div>
            <div class="result-info">
                ♻️ {info['descripcion']}<br>
                📍 {info['contenedor']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilidades
        st.markdown("### 📊 Probabilidades por Clase")
        
        probs = pred['probabilidades']
        for i, clase_nombre in enumerate(CLASSES):
            info_clase = CLASS_INFO[clase_nombre]
            prob = probs[i]
            
            col_emoji, col_bar = st.columns([0.3, 2])
            with col_emoji:
                st.markdown(f"### {info_clase['emoji']}")
            with col_bar:
                st.markdown(f"**{info_clase['nombre']}**")
                st.progress(prob, text=f"{prob*100:.1f}%")
        
        # Timestamp
        st.caption(f"🕐 Clasificado: {pred['timestamp'].strftime('%H:%M:%S')}")
        
    else:
        # Placeholder
        st.info("""
        👆 Captura o sube una imagen para ver los resultados
        
        El sistema identificará el tipo de residuo y te dirá:
        - ✅ Qué es (cartón, vidrio, metal, papel o plástico)
        - 📊 Con qué confianza
        - ♻️ Si es reciclable
        - 📍 En qué contenedor va
        """)

# ============================================================
# HISTORIAL (OPCIONAL)
# ============================================================

if st.session_state.historial:
    st.markdown("---")
    st.markdown("### 📜 Historial Reciente")
    
    # Mostrar últimas 5 clasificaciones
    for i, item in enumerate(reversed(st.session_state.historial[-5:])):
        info = CLASS_INFO[item['clase']]
        with st.expander(f"{info['emoji']} {info['nombre']} - {item['confianza']*100:.1f}% - {item['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Confianza:** {item['confianza']*100:.1f}%")
            st.write(f"**Clase:** {info['nombre']}")
            st.write(f"**Información:** {info['descripcion']}")
            st.write(f"**Contenedor:** {info['contenedor']}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 1rem;'>
    <p>🌍 <strong>Wally AI</strong> - Clasificación Inteligente de Residuos</p>
    <p>Desarrollado con Python + Streamlit + scikit-learn</p>
    <p>Modelo SVM con ~87% de precisión</p>
</div>
""", unsafe_allow_html=True)