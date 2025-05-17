import cv2
import torch
import torchvision.transforms as transforms
from model import cargar_modelo
from PIL import Image
import os

# Config
VIDEO_PATH = "/data/video_prueba.mp4"  # Reemplaza por tu archivo
MODELO_PATH = "/app/modelos/simpson_modelo.pth"
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases (ajústalas según las carpetas de tu dataset)
CLASES = sorted(os.listdir("/data/simpsons_dataset"))  # Asume que están montadas

# Cargar modelo
modelo = cargar_modelo(MODELO_PATH, num_classes=len(CLASES), device=DEVICE)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Leer video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir frame a PIL para aplicar transformaciones
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagen)
    entrada = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predicción
    with torch.no_grad():
        salida = modelo(entrada)
        pred = torch.argmax(salida, dim=1).item()
        clase_predicha = CLASES[pred]

    print(f"[Frame {frame_id}] → {clase_predicha}")
    frame_id += 1

cap.release()
print("✅ Análisis completo.")
