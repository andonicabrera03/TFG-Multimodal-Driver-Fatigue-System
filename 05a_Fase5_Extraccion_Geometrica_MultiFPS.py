"""
05a_Fase5_Extraccion_Geometrica_MultiFPS.py

Autor: Andoni Cabrera Fernández

Script para la extracción offline de características geométricas faciales.
Utiliza el modelo Face Landmarker de MediaPipe para localizar puntos de referencia 
(landmarks) en 3D y calcular las métricas Eye Aspect Ratio (EAR) y Mouth Aspect 
Ratio (MAR) por fotograma.

Los resultados se calculan y guardan para distintas frecuencias de muestreo 
(30, 15, 5 y 1 FPS) en archivos .pkl para su posterior evaluación.
"""

import cv2
import os
import glob
import math
import numpy as np
import pickle
import urllib.request
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =========================================================================
ruta_base_dataset = r"D:\TFG_Fatiga_Andoni\3_Datasets\UTA-RLDD\Videos_Originales\UTA Real-Life Drowsiness Dataset"
# Se guardan los PKL geométricos en una carpeta separada para no sobreescribir los de la red neuronal
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\Resultados_Geometria_PKL"

os.makedirs(ruta_salida_base, exist_ok=True)

# =========================================================================
# 2. INICIALIZACIÓN DE MEDIAPIPE FACE LANDMARKER
# =========================================================================
ruta_mp_task = 'face_landmarker.task'
if not os.path.exists(ruta_mp_task):
    print("Descargando modelo base de MediaPipe (face_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, ruta_mp_task)

print("Inicializando modelo Face Landmarker de MediaPipe...")
base_options = python.BaseOptions(model_asset_path=ruta_mp_task)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False, # Desactivado para optimizar rendimiento
    output_facial_transformation_matrixes=False
)
detector_landmarks = vision.FaceLandmarker.create_from_options(options)
print("Modelo inicializado correctamente.\n")

# =========================================================================
# 3. CÁLCULO DE MÉTRICAS GEOMÉTRICAS
# =========================================================================
def distancia_puntos_3d(p1, p2, w, h):
    """Calcula la distancia euclidiana en 3D entre dos puntos."""
    x1, y1, z1 = p1.x * w, p1.y * h, p1.z * w
    x2, y2, z2 = p2.x * w, p2.y * h, p2.z * w
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calcular_ear_3d(landmarks, w, h):
    """
    Cálculo del Eye Aspect Ratio (EAR) en 3D utilizando 6 puntos faciales por ojo.
    Basado en la formulación de Soukupová y Čech (2016).
    """
    # Ojo Izquierdo
    horiz_izq = distancia_puntos_3d(landmarks[33], landmarks[133], w, h)
    if horiz_izq > 0:
        v1_izq = distancia_puntos_3d(landmarks[160], landmarks[144], w, h)
        v2_izq = distancia_puntos_3d(landmarks[158], landmarks[153], w, h)
        ear_izq = (v1_izq + v2_izq) / (2.0 * horiz_izq)
    else:
        ear_izq = 0.0

    # Ojo Derecho
    horiz_der = distancia_puntos_3d(landmarks[362], landmarks[263], w, h)
    if horiz_der > 0:
        v1_der = distancia_puntos_3d(landmarks[385], landmarks[380], w, h)
        v2_der = distancia_puntos_3d(landmarks[387], landmarks[373], w, h)
        ear_der = (v1_der + v2_der) / (2.0 * horiz_der)
    else:
        ear_der = 0.0

    return (ear_izq + ear_der) / 2.0

def calcular_mar_3d(landmarks, w, h):
    """
    Cálculo del Mouth Aspect Ratio (MAR) en 3D utilizando 8 puntos faciales.
    Basado en la formulación de Assari et al. (2023).
    """
    # 1. Distancia horizontal (comisuras exteriores m1 y m5)
    m1, m5 = 61, 291
    horiz = distancia_puntos_3d(landmarks[m1], landmarks[m5], w, h)
    
    # Prevenir división por cero
    if horiz <= 0:
        return 0.0

    # 2. Distancias verticales (labio superior exterior vs inferior exterior)
    m2, m8 = 37, 84    # Par vertical izquierdo
    m3, m7 = 0, 17     # Par vertical central
    m4, m6 = 267, 314  # Par vertical derecho
    
    v1 = distancia_puntos_3d(landmarks[m2], landmarks[m8], w, h)
    v2 = distancia_puntos_3d(landmarks[m3], landmarks[m7], w, h)
    v3 = distancia_puntos_3d(landmarks[m4], landmarks[m6], w, h)
    
    # 3. Cálculo del MAR
    mar = (v1 + v2 + v3) / (2.0 * horiz)
    
    return mar

# =========================================================================
# 4. PIPELINE DE EXTRACCIÓN PRINCIPAL
# =========================================================================
datos_30fps = []
datos_15fps = []
datos_5fps = []
datos_1fps = [] 

total_videos_procesados = 0
total_frames_analizados = 0

print("Iniciando extracción de características geométricas (Multi-FPS)...\n")

carpetas_fold = sorted([f for f in os.listdir(ruta_base_dataset) if f.startswith("Fold")])

for fold in carpetas_fold:
    ruta_fold = os.path.join(ruta_base_dataset, fold)
    carpetas_sujetos = sorted([s for s in os.listdir(ruta_fold) if os.path.isdir(os.path.join(ruta_fold, s))])
    
    for sujeto in carpetas_sujetos:
        ruta_sujeto_origen = os.path.join(ruta_fold, sujeto)
        videos = glob.glob(os.path.join(ruta_sujeto_origen, "*.*"))
        videos = [v for v in videos if v.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        for ruta_video in videos:
            nombre_video = os.path.basename(ruta_video) 
            clase_fatiga = nombre_video.split('.')[0]   
            
            try:
                clase_real = int(clase_fatiga)
            except ValueError:
                clase_real = -1
                
            print(f"Procesando: {fold} | Sujeto {sujeto} | Video {nombre_video} ...")
            
            # =================================================================
            # EXTRACCIÓN DE CARACTERÍSTICAS
            # =================================================================
            cap = cv2.VideoCapture(ruta_video)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break 
                
                # Inferencia a 30 FPS (Se procesa cada frame)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                resultados = detector_landmarks.detect(mp_image)

                if resultados.face_landmarks:
                    landmarks = resultados.face_landmarks[0]
                    
                    ear_actual = calcular_ear_3d(landmarks, w, h)
                    mar_actual = calcular_mar_3d(landmarks, w, h)

                    # Almacenamiento de métricas del fotograma actual
                    dato_frame = {
                        'video': f"{sujeto}_{nombre_video}", # ID único para trazabilidad
                        'clase_real': clase_real, 
                        'frame_idx': frame_idx,
                        'ear': ear_actual,
                        'mar': mar_actual
                    }

                    # =====================================================
                    # SUBMUESTREO TEMPORAL (30, 15, 5 y 1 FPS)
                    # =====================================================
                    datos_30fps.append(dato_frame)               
                    if frame_idx % 2 == 0: datos_15fps.append(dato_frame) 
                    if frame_idx % 6 == 0: datos_5fps.append(dato_frame)  
                    if frame_idx % 30 == 0: datos_1fps.append(dato_frame) # <-- Extrae 1 frame por cada 30
                    
                    total_frames_analizados += 1

                frame_idx += 1
            
            cap.release()
            total_videos_procesados += 1

# =========================================================================
# 5. GUARDADO DE RESULTADOS (PICKLE)
# =========================================================================
print("\nGuardando resultados en archivos PKL...")
with open(os.path.join(ruta_salida_base, 'geometria_30fps.pkl'), 'wb') as f: pickle.dump(datos_30fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_15fps.pkl'), 'wb') as f: pickle.dump(datos_15fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_5fps.pkl'), 'wb') as f: pickle.dump(datos_5fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_1fps.pkl'), 'wb') as f: pickle.dump(datos_1fps, f)

print("\n==========================================")
print("¡EXTRACCIÓN FINALIZADA!")
print(f"Vídeos analizados:          {total_videos_procesados}")
print(f"Fotogramas procesados:      {total_frames_analizados}")
print(f"Archivos PKL generados en:  {ruta_salida_base}")
print("==========================================")