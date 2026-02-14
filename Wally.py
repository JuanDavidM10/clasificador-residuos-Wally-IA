"""
╔══════════════════════════════════════════════════════════════╗
║          Wally AI - CLASIFICADOR DE RESIDUOS             ║
║         Versión COMPACTA para pantallas pequeñas             ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import joblib
from skimage.feature import hog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from datetime import datetime
from collections import Counter
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Parámetros del modelo
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
IMG_SIZE = (256, 256)

# Parámetros HOG
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)
HOG_BLOCK_NORM = "L2-Hys"

# Colores para cada clase
CLASS_COLORS = {
    "cardboard": ("#8B4513", "🟤"),
    "glass": ("#4FC3F7", "🔵"),
    "metal": ("#9E9E9E", "⚪"),
    "paper": ("#FFD700", "🟡"),
    "plastic": ("#E91E63", "🔴")
}

CLASS_INFO = {
    "cardboard": ("CARTÓN", "♻️ Reciclable", "Contenedor azul"),
    "glass": ("VIDRIO", "♻️ Reciclable", "Contenedor verde"),
    "metal": ("METAL", "♻️ Reciclable", "Contenedor amarillo"),
    "paper": ("PAPEL", "♻️ Reciclable", "Contenedor azul"),
    "plastic": ("PLÁSTICO", "⚠️ Reciclable", "Contenedor amarillo")
}

# ============================================================
# CLASE PRINCIPAL - VERSIÓN COMPACTA
# ============================================================

class EcoVisionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🌍 Wally AI - Clasificador de Residuos")
        
        # TAMAÑO AJUSTABLE - cambia según tu pantalla
        # Para 1366x768: usa "1200x700"
        # Para 1920x1080: usa "1400x900"
        self.root.geometry("1200x700")
        
        # Variables
        self.camera = None
        self.is_running = False
        self.modelo = None
        self.scaler = None
        self.pca = None
        self.frame_actual = None
        
        # Estadísticas
        self.historial = []
        self.contador_clases = Counter()
        self.total_clasificaciones = 0
        
        # Cargar modelo
        self.cargar_modelo()
        
        # Crear interfaz
        self.crear_interfaz()
        
        # Iniciar
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
    def cargar_modelo(self):
        """Carga los modelos entrenados"""
        try:
            print("📦 Cargando modelos...")
            self.modelo = joblib.load("modelo_final.pkl")
            self.scaler = joblib.load("scaler.pkl")
            self.pca = joblib.load("pca.pkl")
            print("✅ Modelos cargados correctamente")
            return True
        except Exception as e:
            print(f"❌ Error al cargar modelos: {e}")
            return False
    
    def crear_interfaz(self):
        """Crea interfaz compacta"""
        
        # ============================================================
        # HEADER COMPACTO
        # ============================================================
        header_frame = ctk.CTkFrame(self.root, height=60, fg_color="#1a1a1a")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="♻️ Wally AI",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#10B981"
        )
        title_label.pack(side="left", padx=20, pady=10)
        
        estado_text = "🟢 Listo" if self.modelo else "🔴 Error"
        estado_color = "#10B981" if self.modelo else "#EF4444"
        
        self.estado_label = ctk.CTkLabel(
            header_frame,
            text=estado_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=estado_color
        )
        self.estado_label.pack(side="right", padx=20, pady=10)
        
        # ============================================================
        # CONTENEDOR PRINCIPAL CON SCROLL
        # ============================================================
        
        # Frame scrollable
        main_scroll = ctk.CTkScrollableFrame(
            self.root,
            fg_color="transparent"
        )
        main_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # ============================================================
        # PANEL DE CÁMARA (MÁS COMPACTO)
        # ============================================================
        camera_frame = ctk.CTkFrame(main_scroll, fg_color="#1E293B")
        camera_frame.pack(fill="x", pady=(0, 10))
        
        camera_title = ctk.CTkLabel(
            camera_frame,
            text="📹 Cámara",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#10B981"
        )
        camera_title.pack(pady=10)
        
        # Canvas más pequeño
        self.canvas = ctk.CTkCanvas(
            camera_frame,
            width=640,
            height=480,
            bg="#0F172A",
            highlightthickness=2,
            highlightbackground="#10B981"
        )
        self.canvas.pack(padx=15, pady=(0, 15))
        
        self.canvas.create_text(
            320, 240,
            text="📹\nCámara Detenida\n\nPresiona 'Iniciar'",
            font=("Arial", 18),
            fill="#64748B",
            justify="center",
            tags="placeholder"
        )
        
        # ============================================================
        # CONTROLES EN LÍNEA
        # ============================================================
        controls_frame = ctk.CTkFrame(main_scroll, fg_color="#1E293B")
        controls_frame.pack(fill="x", pady=(0, 10))
        
        controls_title = ctk.CTkLabel(
            controls_frame,
            text="🎮 Controles",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        controls_title.pack(pady=10)
        
        buttons_container = ctk.CTkFrame(controls_frame, fg_color="transparent")
        buttons_container.pack(pady=(0, 15))
        
        self.btn_iniciar = ctk.CTkButton(
            buttons_container,
            text="▶️ Iniciar",
            command=self.iniciar_camara,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669"
        )
        self.btn_iniciar.pack(side="left", padx=5)
        
        self.btn_clasificar = ctk.CTkButton(
            buttons_container,
            text="📸 Clasificar",
            command=self.clasificar_frame,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            state="disabled"
        )
        self.btn_clasificar.pack(side="left", padx=5)
        
        self.btn_detener = ctk.CTkButton(
            buttons_container,
            text="⏹️ Detener",
            command=self.detener_camara,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            state="disabled"
        )
        self.btn_detener.pack(side="left", padx=5)
        
        self.btn_limpiar = ctk.CTkButton(
            buttons_container,
            text="🗑️ Limpiar",
            command=self.limpiar_estadisticas,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#64748B",
            hover_color="#475569"
        )
        self.btn_limpiar.pack(side="left", padx=5)
        
        # ============================================================
        # RESULTADO
        # ============================================================
        result_frame = ctk.CTkFrame(main_scroll, fg_color="#1E293B")
        result_frame.pack(fill="x", pady=(0, 10))
        
        result_title = ctk.CTkLabel(
            result_frame,
            text="🎯 Resultado",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#10B981"
        )
        result_title.pack(pady=10)
        
        self.result_container = ctk.CTkFrame(result_frame, fg_color="#0F172A")
        self.result_container.pack(padx=15, pady=(0, 15), fill="x")
        
        self.result_icon = ctk.CTkLabel(
            self.result_container,
            text="🎯",
            font=ctk.CTkFont(size=60)
        )
        self.result_icon.pack(pady=(15, 5))
        
        self.result_class = ctk.CTkLabel(
            self.result_container,
            text="Esperando...",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#94A3B8"
        )
        self.result_class.pack(pady=5)
        
        self.result_confidence = ctk.CTkLabel(
            self.result_container,
            text="",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#10B981"
        )
        self.result_confidence.pack(pady=5)
        
        self.result_info = ctk.CTkLabel(
            self.result_container,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#64748B"
        )
        self.result_info.pack(pady=(5, 15))
        
        # ============================================================
        # PROBABILIDADES
        # ============================================================
        prob_frame = ctk.CTkFrame(main_scroll, fg_color="#1E293B")
        prob_frame.pack(fill="x", pady=(0, 10))
        
        prob_title = ctk.CTkLabel(
            prob_frame,
            text="📊 Probabilidades",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        prob_title.pack(pady=10)
        
        self.prob_container = ctk.CTkFrame(prob_frame, fg_color="#0F172A")
        self.prob_container.pack(padx=15, pady=(0, 15), fill="x")
        
        self.prob_bars = {}
        self.prob_labels = {}
        
        for clase in CLASSES:
            emoji, _ = CLASS_COLORS[clase]
            nombre, _, _ = CLASS_INFO[clase]
            
            class_frame = ctk.CTkFrame(self.prob_container, fg_color="transparent")
            class_frame.pack(fill="x", padx=10, pady=5)
            
            header = ctk.CTkLabel(
                class_frame,
                text=f"{nombre}",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w",
                width=80
            )
            header.pack(side="left", padx=(0, 8))
            
            progress = ctk.CTkProgressBar(
                class_frame,
                width=350,
                height=16,
                progress_color=emoji
            )
            progress.pack(side="left", padx=(0, 8))
            progress.set(0)
            
            percent_label = ctk.CTkLabel(
                class_frame,
                text="0.0%",
                font=ctk.CTkFont(size=11, weight="bold"),
                width=50
            )
            percent_label.pack(side="left")
            
            self.prob_bars[clase] = progress
            self.prob_labels[clase] = percent_label
        
        # ============================================================
        # ESTADÍSTICAS COMPACTAS
        # ============================================================
        stats_frame = ctk.CTkFrame(main_scroll, fg_color="#1E293B")
        stats_frame.pack(fill="x", pady=(0, 10))
        
        stats_title = ctk.CTkLabel(
            stats_frame,
            text="📈 Estadísticas",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_title.pack(pady=10)
        
        stats_grid = ctk.CTkFrame(stats_frame, fg_color="#0F172A")
        stats_grid.pack(padx=15, pady=(0, 15), fill="x")
        
        # Total
        total_frame = ctk.CTkFrame(stats_grid, fg_color="#1E293B")
        total_frame.pack(fill="x", padx=10, pady=8)
        
        ctk.CTkLabel(
            total_frame,
            text="🎯 Total:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=8)
        
        self.total_label = ctk.CTkLabel(
            total_frame,
            text="0",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#10B981"
        )
        self.total_label.pack(side="right", padx=8)
        
        # Por clase
        self.stats_labels = {}
        
        for clase in CLASSES:
            emoji, icono = CLASS_COLORS[clase]
            nombre, _, _ = CLASS_INFO[clase]
            
            class_stat_frame = ctk.CTkFrame(stats_grid, fg_color="#1E293B")
            class_stat_frame.pack(fill="x", padx=10, pady=3)
            
            ctk.CTkLabel(
                class_stat_frame,
                text=f"{icono} {nombre}:",
                font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=8)
            
            count_label = ctk.CTkLabel(
                class_stat_frame,
                text="0",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=emoji
            )
            count_label.pack(side="right", padx=8)
            
            self.stats_labels[clase] = count_label
    
    def iniciar_camara(self):
        """Inicia la cámara"""
        if not self.is_running:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                self.mostrar_error("No se pudo acceder a la cámara.\nVerifica que esté conectada.")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            
            self.btn_iniciar.configure(state="disabled")
            self.btn_clasificar.configure(state="normal")
            self.btn_detener.configure(state="normal")
            self.estado_label.configure(text="🟢 Activa", text_color="#10B981")
            
            self.actualizar_frame()
    
    def detener_camara(self):
        """Detiene la cámara"""
        if self.is_running:
            self.is_running = False
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.btn_iniciar.configure(state="normal")
            self.btn_clasificar.configure(state="disabled")
            self.btn_detener.configure(state="disabled")
            self.estado_label.configure(text="🟡 Detenida", text_color="#F59E0B")
            
            self.canvas.delete("all")
            self.canvas.create_text(
                320, 240,
                text="📹\nCámara Detenida\n\nPresiona 'Iniciar'",
                font=("Arial", 18),
                fill="#64748B",
                justify="center"
            )
    
    def actualizar_frame(self):
        """Actualiza el frame de la cámara"""
        if self.is_running and self.camera:
            ret, frame = self.camera.read()
            
            if ret:
                self.frame_actual = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")
                self.canvas.create_image(320, 240, image=imgtk)
                self.canvas.image = imgtk
                
            self.root.after(10, self.actualizar_frame)
    
    def clasificar_frame(self):
        """Clasifica el frame actual"""
        if self.frame_actual is None:
            self.mostrar_error("No hay frame para clasificar")
            return
        
        if self.modelo is None:
            self.mostrar_error("Modelo no cargado")
            return
        
        self.btn_clasificar.configure(state="disabled", text="⏳ Procesando...")
        
        thread = threading.Thread(target=self._clasificar_thread)
        thread.daemon = True
        thread.start()
    
    def _clasificar_thread(self):
        """Thread de clasificación"""
        try:
            img_resized = cv2.resize(self.frame_actual, IMG_SIZE)
            
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            features = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm=HOG_BLOCK_NORM,
                visualize=False,
                feature_vector=True
            )
            
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            
            prediction = self.modelo.predict(features_pca)[0]
            probabilities = self.modelo.predict_proba(features_pca)[0]
            
            self.root.after(0, self.actualizar_resultado, prediction, probabilities)
            
        except Exception as e:
            self.root.after(0, self.mostrar_error, f"Error:\n{e}")
        finally:
            self.root.after(0, lambda: self.btn_clasificar.configure(
                state="normal", text="📸 Clasificar"
            ))
    
    def actualizar_resultado(self, prediction, probabilities):
        """Actualiza UI con resultado"""
        clase = CLASSES[prediction]
        confianza = probabilities[prediction]
        
        nombre, descripcion, contenedor = CLASS_INFO[clase]
        color, icono = CLASS_COLORS[clase]
        
        self.result_icon.configure(text=icono)
        self.result_class.configure(text=nombre, text_color=color)
        self.result_confidence.configure(text=f"{confianza*100:.1f}%")
        self.result_info.configure(text=f"{descripcion}\n{contenedor}")
        
        for i, clase_nombre in enumerate(CLASSES):
            prob = probabilities[i]
            self.prob_bars[clase_nombre].set(prob)
            self.prob_labels[clase_nombre].configure(text=f"{prob*100:.1f}%")
        
        self.contador_clases[clase] += 1
        self.total_clasificaciones += 1
        
        self.total_label.configure(text=str(self.total_clasificaciones))
        self.stats_labels[clase].configure(text=str(self.contador_clases[clase]))
        
        print(f"✅ {nombre} ({confianza*100:.1f}%)")
    
    def limpiar_estadisticas(self):
        """Limpia estadísticas"""
        self.contador_clases = Counter()
        self.total_clasificaciones = 0
        
        self.total_label.configure(text="0")
        for clase in CLASSES:
            self.stats_labels[clase].configure(text="0")
            self.prob_bars[clase].set(0)
            self.prob_labels[clase].configure(text="0.0%")
        
        self.result_icon.configure(text="🎯")
        self.result_class.configure(text="Esperando...", text_color="#94A3B8")
        self.result_confidence.configure(text="")
        self.result_info.configure(text="")
        
        print("🗑️ Limpiado")
    
    def mostrar_error(self, mensaje):
        """Muestra error"""
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")
        error_window.geometry("400x180")
        error_window.transient(self.root)
        error_window.grab_set()
        
        ctk.CTkLabel(
            error_window,
            text="❌ Error",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#EF4444"
        ).pack(pady=15)
        
        ctk.CTkLabel(
            error_window,
            text=mensaje,
            font=ctk.CTkFont(size=13),
            wraplength=350
        ).pack(pady=10)
        
        ctk.CTkButton(
            error_window,
            text="Cerrar",
            command=error_window.destroy,
            fg_color="#EF4444",
            hover_color="#DC2626"
        ).pack(pady=15)
    
    def cerrar_aplicacion(self):
        """Cierra app"""
        self.detener_camara()
        self.root.destroy()
    
    def run(self):
        """Ejecuta app"""
        self.root.mainloop()


# ============================================================
# INICIO
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ♻️  WALLY AI - VERSIÓN COMPACTA  ♻️")
    print("="*60)
    print("\n🚀 Iniciando...\n")
    
    archivos_necesarios = ["modelo_final.pkl", "scaler.pkl", "pca.pkl"]
    archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]
    
    if archivos_faltantes:
        print("❌ ERROR: Archivos del modelo no encontrados:")
        for archivo in archivos_faltantes:
            print(f"   - {archivo}")
        print("\n💡 Colócalos en el mismo directorio.\n")
        input("Presiona Enter para salir...")
    else:
        app = EcoVisionApp()
        app.run()