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
# FASE 5a: EXTRACCIÓN TOPOLÓGICA 3D OFFLINE (EAR y MAR) -> Multi-FPS
# =========================================================================
# Autor: Andoni Cabrera Fernández
# Descripción: Inferencia geométrica explícita (TinyML en CPU). Abandona 
#              la inferencia de píxeles (Fase 4) por distancias espaciales 
#              euclidianas en 3D (Invarianza Rotacional).
#              Serializa EAR y MAR crudos en archivos PKL.
# =========================================================================

# 1. Configuración de Rutas (Estructura idéntica a las fases anteriores)
ruta_base_dataset = r"D:\TFG_Fatiga_Andoni\3_Datasets\UTA-RLDD\Videos_Originales\UTA Real-Life Drowsiness Dataset"
# Podemos guardar los PKL geométricos en una carpeta separada para no pisar los de la red neuronal
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\Resultados_Geometria_PKL"

os.makedirs(ruta_salida_base, exist_ok=True)

# 2. Inicialización de MediaPipe Face Landmarker (Local / CPU)
ruta_mp_task = 'face_landmarker.task'
if not os.path.exists(ruta_mp_task):
    print("Descargando modelo de MediaPipe (face_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, ruta_mp_task)

print("Iniciando Motor Geométrico MediaPipe en CPU...")
base_options = python.BaseOptions(model_asset_path=ruta_mp_task)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False, # Desactivado para maximizar rendimiento
    output_facial_transformation_matrixes=False
)
detector_landmarks = vision.FaceLandmarker.create_from_options(options)
print("Motor matemático inicializado correctamente.\n")

# 3. Módulo Matemático: Distancia Euclidiana 3D
def distancia_puntos_3d(p1, p2, w, h):
    """Calcula la distancia Euclidiana en 3D para garantizar Invarianza Rotacional"""
    x1, y1, z1 = p1.x * w, p1.y * h, p1.z * w
    x2, y2, z2 = p2.x * w, p2.y * h, p2.z * w
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calcular_ear_3d(landmarks, w, h):
    """
    Eye Aspect Ratio (EAR) tridimensional utilizando topología de 6 puntos.
    Basado en la formulación geométrica de Soukupová y Čech (2016).
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
    Mouth Aspect Ratio (MAR) tridimensional utilizando topología de 8 puntos.
    Basado en la formulación geométrica de Assari et al. (2023).
    """
    # 1. Distancia horizontal (comisuras exteriores m1 y m5)
    m1, m5 = 61, 291
    horiz = distancia_puntos_3d(landmarks[m1], landmarks[m5], w, h)
    
    # Prevenir división por cero en caso de colapso topológico
    if horiz <= 0:
        return 0.0

    # 2. Distancias verticales (labio superior exterior vs inferior exterior)
    m2, m8 = 37, 84    # Par vertical izquierdo
    m3, m7 = 0, 17     # Par vertical central
    m4, m6 = 267, 314  # Par vertical derecho
    
    v1 = distancia_puntos_3d(landmarks[m2], landmarks[m8], w, h)
    v2 = distancia_puntos_3d(landmarks[m3], landmarks[m7], w, h)
    v3 = distancia_puntos_3d(landmarks[m4], landmarks[m6], w, h)
    
    # 3. Formulación matemática ponderada
    mar = (v1 + v2 + v3) / (2.0 * horiz)
    
    return mar

# 4. Pipeline de Extracción Principal
datos_30fps = []
datos_15fps = []
datos_5fps = []
datos_1fps = [] # <-- Nueva lista para 1 FPS

total_videos_procesados = 0
total_frames_analizados = 0

print("Iniciando Pipeline de Extracción Topológica 3D (Multi-FPS)...\n")

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
                
            print(f"Procesando Geometría: {fold} | Sujeto {sujeto} | Video {nombre_video} ...")
            
            # =================================================================
            # INFERENCIA GEOMÉTRICA ESPACIAL
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

                    # Diccionario de coordenadas del fotograma actual
                    dato_frame = {
                        'video': f"{sujeto}_{nombre_video}", # ID único para trazabilidad
                        'clase_real': clase_real, 
                        'frame_idx': frame_idx,
                        'ear': ear_actual,
                        'mar': mar_actual
                    }

                    # =====================================================
                    # MULTIPLEXACIÓN TEMPORAL (30, 15, 5 y 1 FPS)
                    # =====================================================
                    datos_30fps.append(dato_frame)               
                    if frame_idx % 2 == 0: datos_15fps.append(dato_frame) 
                    if frame_idx % 6 == 0: datos_5fps.append(dato_frame)  
                    if frame_idx % 30 == 0: datos_1fps.append(dato_frame) # <-- Extrae 1 frame por cada 30
                    
                    total_frames_analizados += 1

                frame_idx += 1
            
            cap.release()
            total_videos_procesados += 1

# =================================================================
# SERIALIZACIÓN FINAL EN ARCHIVOS PICKLE
# =================================================================
print("\nSerializando coordenadas trigonométricas en archivos PKL...")
with open(os.path.join(ruta_salida_base, 'geometria_30fps.pkl'), 'wb') as f: pickle.dump(datos_30fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_15fps.pkl'), 'wb') as f: pickle.dump(datos_15fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_5fps.pkl'), 'wb') as f: pickle.dump(datos_5fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_1fps.pkl'), 'wb') as f: pickle.dump(datos_1fps, f)

print("\n==========================================")
print("¡PIPELINE DE EXTRACCIÓN GEOMÉTRICA FINALIZADO!")
print(f"Vídeos analizados en esta sesión: {total_videos_procesados}")
print(f"Fotogramas calculados (Euclides): {total_frames_analizados}")
print(f"Archivos PKL generados en:        {ruta_salida_base}")
print("==========================================")