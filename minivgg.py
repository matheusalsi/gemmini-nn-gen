import torch
import torch.nn as nn
import torchvision.models as models

# 1. Definir a arquitetura da MiniVGGNet
class MiniVGGNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(MiniVGGNet, self).__init__()
        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # Tamanho ajustado para entrada 28x28
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Instanciar o modelo
num_classes = 10 # Exemplo para o dataset MNIST
model = MiniVGGNet(num_channels=1, num_classes=num_classes)

# 3. Criar um "dummy input" com o tamanho esperado (e.g., para MNIST)
# Batch size=1, 1 canal de cor, 28x28 de resolução
dummy_input = torch.randn(1, 1, 28, 28)

# 4. Exportar o modelo para ONNX
torch.onnx.export(model, dummy_input, "minivggnet.onnx", verbose=True, input_names=['input'], output_names=['output'])

print("minivggnet.onnx criado com sucesso!")