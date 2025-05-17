import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpsonCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpsonCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # ajusta según tu imagen si no es 128x128
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, H/4, W/4]
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Cargar modelo desde archivo .pth
def cargar_modelo(ruta_pesos, num_classes=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = SimpsonCNN(num_classes=num_classes).to(device)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.eval()
    return modelo
