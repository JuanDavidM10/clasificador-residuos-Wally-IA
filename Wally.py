import streamlit as st
import numpy as np
import cv2
import joblib
import time
from PIL import Image
from skimage.feature import hog

# ==============================
# CONFIGURACIÓN
# ==============================

IMG_SIZE = (256, 256)

HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)
HOG_BLOCK_NORM = "L2-Hys"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

st.set_page_config(page_title="♻️ Wally AI - Clasificador Inteligente", layout="wide")

# ==============================
# CARGAR MODELOS
# ==============================

@st.cache_resource
def load_models():
    model = joblib.load("modelo_final.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return model, scaler, pca

model, scaler, pca = load_models()

# ==============================
# INTERFAZ
# ==============================

st.title("♻️ Wally AI - Clasificador Inteligente de Residuos")
st.markdown("### Aplicación en Tiempo Real - SVM + HOG + PCA")

option = st.radio(
    "Seleccione método:",
    ["📷 Cámara", "🖼 Subir Imagen"],
    horizontal=True
)

image = None

if option == "📷 Cámara":
    image = st.camera_input("Tome una foto")

elif option == "🖼 Subir Imagen":
    uploaded = st.file_uploader("Suba una imagen", type=["jpg","png","jpeg"])
    if uploaded:
        image = uploaded

# ==============================
# PROCESAMIENTO
# ==============================

if image is not None:

    img = Image.open(image)
    img = np.array(img)

    st.image(img, caption="Imagen original", width=300)

    # Resize igual que entrenamiento
    img_resized = cv2.resize(img, IMG_SIZE)

    # Convertir a gris
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Extraer HOG
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True
    )

    hog_features = hog_features.reshape(1, -1)

    # Predicción
    start = time.time()

    scaled = scaler.transform(hog_features)
    pca_features = pca.transform(scaled)
    prediction = model.predict(pca_features)
    probabilities = model.predict_proba(pca_features)

    end = time.time()

    predicted_class = CLASSES[prediction[0]]
    confidence = np.max(probabilities) * 100
    inference_time = end - start

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🧠 Predicción: {predicted_class.upper()}")
        st.write(f"📊 Confianza: {confidence:.2f}%")
        st.write(f"⏱ Tiempo de inferencia: {inference_time:.4f} segundos")

    with col2:
        st.subheader("Probabilidades por clase")
        for i, clase in enumerate(CLASSES):
            st.progress(float(probabilities[0][i]))
            st.write(f"{clase}: {probabilities[0][i]*100:.2f}%")

st.markdown("---")
st.markdown("Proyecto T4 - Juan David M 🚀")

