import streamlit as st
import cv2
import numpy as np
import rembg
import re
from PIL import Image
from ultralytics import YOLO
from process.ocr_extraction.main_ocr import TextExtraction
from threading import Thread
import sqlite3
import base64
from datetime import datetime
import io
import os

st.set_page_config(
    page_title="Detección de Placa en Tiempo Real con YOLOv8 y OCR",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Parámetros de login
VALID_USERNAME = "usuariodeprueba"
VALID_PASSWORD = "lamalapayerson01"

# Función de autenticación
def login(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD

# Verificar el estado de sesión
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Pantalla de inicio de sesión
if not st.session_state.authenticated:
    st.title("Iniciar Sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Iniciar Sesión")
    
    if login_button:
        if login(username, password):
            st.session_state.authenticated = True
            st.success("Inicio de sesión exitoso")
            st.rerun()  # Recargar la interfaz para mostrar la aplicación
        else:
            st.error("Usuario o contraseña incorrectos")
    st.stop()  # Detener aquí si no está autenticado

# --- Código de la aplicación principal después del inicio de sesión ---
# Define el encabezado de la aplicación
st.header('Detección de Placa de Vehículo en Tiempo Real con YOLOv8 y Reconocimiento OCR')
st.warning("Haz clic en 'Activar cámara' para iniciar la detección de placas en vivo.", icon=":material/warning:")

# Ruta para guardar la base de datos en la carpeta db
DB_PATH = "db/detected_plates.db"

# Crear la carpeta db si no existe
os.makedirs("db", exist_ok=True)

# Inicializar la base de datos SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_image_base64 TEXT,
            detection_date TEXT
        )
    """)
    conn.commit()
    conn.close()

# Guardar una detección en la base de datos
def save_detection(plate_text, plate_image, detection_date):
    # Convertir la imagen a base64
    buffered = io.BytesIO()
    plate_image.save(buffered, format="PNG")
    plate_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Insertar el registro en la base de datos
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO detected_plates (plate_text, plate_image_base64, detection_date) VALUES (?, ?, ?)",
                   (plate_text, plate_image_base64, detection_date))
    conn.commit()
    conn.close()

# Inicializar la base de datos al inicio
init_db()

# Clase para capturar video en vivo
class RunCamera:
    def __init__(self, src=0):
        self.src = src
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False
        (self.grabbed, self.frameR) = self.capture.read()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self
    
    def get(self):
        while not self.stopped:
            (self.grabbed, self.frameR) = self.capture.read()
            if not self.grabbed:
                self.stop()
    
    def stop(self):
        self.stopped = True
        self.capture.release()

# Cargar el modelo YOLOv8 personalizado desde la ruta especificada
modelo_yolo = YOLO('C:/Users/David/Desktop/plate_segmentation.pt')  # Ruta actualizada

# Instanciar el procesador de OCR optimizado
ocr_processor = TextExtraction()

# Función para alinear el texto al formato de placa colombiana
def formatear_texto_placa(texto):
    texto = re.sub(r'[^A-Za-z0-9]', '', texto)  # Eliminar caracteres no alfanuméricos
    match = re.match(r'([A-Za-z]{3})([0-9]{3})', texto)
    if match:
        return match.group(1).upper() + match.group(2)
    else:
        letras = ''.join([c for c in texto if c.isalpha()])[:3].upper()
        numeros = ''.join([c for c in texto if c.isdigit()])[:3]
        if len(letras) == 3 and len(numeros) == 3:
            return letras + numeros
        else:
            return "Formato no reconocido"

# Configuración de estado para el manejo de la cámara
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Colocar ambos botones en una fila estrecha
button_col1, button_col2, _ = st.columns([1, 1, 8])
with button_col1:
    activar = st.button("Activar cámara")
with button_col2:
    desactivar = st.button("Desactivar cámara")

# Manejar el estado de los botones
if activar and not st.session_state.camera_active:
    st.session_state.run_camera = RunCamera(src=0).start()
    st.session_state.camera_active = True

if desactivar and st.session_state.camera_active:
    st.session_state.run_camera.stop()
    st.session_state.run_camera = None
    st.session_state.camera_active = False

# Colocar el video en vivo en una columna y el crop de la placa y el texto en otra columna alineada a la izquierda
left_col, right_col, _ = st.columns([3, 2, 1])

# Contenedores para el video en vivo, la placa detectada y el texto del OCR
with left_col:
    video_container = st.empty()
with right_col:
    plate_container = st.empty()
    text_container = st.empty()

# Procesamiento en vivo si la cámara está activa
if st.session_state.camera_active and st.session_state.run_camera is not None:
    try:
        while st.session_state.camera_active:
            # Obtener el frame actual
            frame = st.session_state.run_camera.frameR
            if frame is None:
                continue
            
            # Redimensionar el frame a 640x480
            frame = cv2.resize(frame, (640, 480))
            
            # Aplicar eliminación de fondo
            output_array = rembg.remove(frame)
            
            if output_array.shape[2] == 4:
                output_array = cv2.cvtColor(output_array, cv2.COLOR_BGRA2BGR)
            
            output_gray = cv2.cvtColor(output_array, cv2.COLOR_BGR2GRAY)
            
            # Detección de placa usando YOLO
            results = modelo_yolo.predict(output_array, imgsz=640)
            detecciones = results[0].boxes
            
            placa_detectada = None
            for box in detecciones:
                class_id = int(box.cls[0])
                if modelo_yolo.names[class_id] == 'number plate':
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    placa_detectada = output_gray[y_min:y_max, x_min:x_max]
                    break

            # Mostrar el flujo de la cámara en tiempo real con el bounding box
            video_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Video en vivo", width=640)
            
            # Mostrar la placa detectada y procesar OCR
            if placa_detectada is not None:
                placa_img_pil = Image.fromarray(placa_detectada)  # Convertir a formato PIL para almacenar
                plate_container.image(placa_img_pil, caption="Placa Detectada", width=320)
                
                # Aplicar OCR
                texto_ocr = ocr_processor.text_extraction(placa_detectada)
                texto_placa = formatear_texto_placa(texto_ocr)
                
                # Guardar en la base de datos
                detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_detection(texto_placa, placa_img_pil, detection_date)
            else:
                texto_placa = "No se detectó ninguna placa"
                plate_container.image(np.zeros((100, 320, 3), dtype=np.uint8), caption="Placa Detectada", width=320)

            # Actualizar el recuadro de texto detectado
            text_container.subheader("Texto Detectado en la Placa")
            text_container.write(f"**Placa:** {texto_placa}")

    except Exception as e:
        st.error(f"Error al procesar el flujo en vivo: {e}")
        st.session_state.run_camera.stop()
        st.session_state.run_camera = None
        st.session_state.camera_active = False
