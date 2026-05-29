import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CARTELLA_DATI = 'dataset-resized'
# TIPO_MODELLO = 'resnet'
# PERCORSO_MODELLO = 'modello_resnet18_trashnet.pth'

TIPO_MODELLO = 'cnn'
PERCORSO_MODELLO = 'modello_trashnet.pth'

def ottieni_test_loader():
    test_transforms = transforms.Compose([
        transforms.CenterCrop(384),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_intero = datasets.ImageFolder(root=CARTELLA_DATI)
    _, indici_test = train_test_split(
        list(range(len(dataset_intero))), test_size=0.20, stratify=dataset_intero.targets, random_state=42
    )
    test_dataset = Subset(datasets.ImageFolder(CARTELLA_DATI, transform=test_transforms), indici_test)
    return DataLoader(test_dataset, batch_size=32, shuffle=False), dataset_intero.classes

def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, nomi_classi = ottieni_test_loader()
    
    if TIPO_MODELLO == 'resnet':
        modello = models.resnet18()
        modello.fc = nn.Linear(modello.fc.in_features, 6)
    elif TIPO_MODELLO == 'cnn':
        from CNN import TrashNetCNN
        modello = TrashNetCNN()
    else:
        return

    modello.load_state_dict(torch.load(PERCORSO_MODELLO, map_location=dispositivo))
    modello = modello.to(dispositivo)
    modello.eval()

    etichette_vere, etichette_predette = [], []
    with torch.no_grad():
        for immagini, etichette in test_loader:
            previsioni = modello(immagini.to(dispositivo))
            _, previsti = torch.max(previsioni, 1)
            etichette_vere.extend(etichette.numpy())
            etichette_predette.extend(previsti.cpu().numpy())

    cm = confusion_matrix(etichette_vere, etichette_predette)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomi_classi, yticklabels=nomi_classi)
    plt.title(f'Matrice di Confusione - {TIPO_MODELLO.upper()}', fontsize=16)
    plt.ylabel('Classe Reale', fontsize=12)
    plt.xlabel('Classe Predetta', fontsize=12)
    plt.tight_layout()
    nome_immagine = f"matrice_confusione_{TIPO_MODELLO}.png"
    plt.savefig(nome_immagine, dpi=300, bbox_inches='tight') # dpi=300 garantisce alta qualità
    print(f"Grafico salvato con successo come immagine: {nome_immagine}")

if __name__ == '__main__':
    main()