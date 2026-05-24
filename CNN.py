import torch
import torch.nn as nn
import torch.nn.functional as F

class TrashNetCNN(nn.Module):
    def __init__(self):
        super(TrashNetCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    modello = TrashNetCNN()
    print(modello)
    
    immagine_finta = torch.randn(1, 3, 224, 224)
    risultato = modello(immagine_finta)
    
    print(f"\nFormato di uscita: {risultato.shape}")