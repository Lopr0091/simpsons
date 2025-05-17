import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpsonCNN

# Configuración
DATA_DIR = "/data/simpsons_dataset"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Cargar dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Clases detectadas
num_classes = len(dataset.classes)
print("Clases:", dataset.classes)
print("Total de imágenes:", len(dataset))

# Modelo
model = SimpsonCNN(num_classes=num_classes).to(DEVICE)

# Pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
print("Entrenando...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{EPOCHS} - Pérdida: {running_loss/len(dataloader):.4f}")

# Guardar el modelo
os.makedirs("/app/modelos", exist_ok=True)
torch.save(model.state_dict(), "/app/modelos/simpson_modelo.pth")
print("✅ Modelo guardado en /app/modelos/simpson_modelo.pth")
