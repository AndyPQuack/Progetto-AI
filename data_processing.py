import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter

train_preprocessing = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_preprocessing = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cartella_dati = r'dataset-resized'

dataset_intero = datasets.ImageFolder(root=cartella_dati)
nomi_classi = dataset_intero.classes
etichette_reali = dataset_intero.targets

indici_train, indici_test = train_test_split(
    list(range(len(dataset_intero))), 
    test_size=0.20, 
    stratify=etichette_reali,
    random_state=42
)

train_dataset = Subset(datasets.ImageFolder(cartella_dati, transform=train_preprocessing), indici_train)
test_dataset = Subset(datasets.ImageFolder(cartella_dati, transform=test_preprocessing), indici_test)

etichette_train = [etichette_reali[i] for i in indici_train]
conteggio_classi = Counter(etichette_train)
pesi_classi = {classe: 1.0 / conteggio for classe, conteggio in conteggio_classi.items()}
pesi_campioni = [pesi_classi[etichetta] for etichetta in etichette_train]
pesi_campioni = torch.DoubleTensor(pesi_campioni)

campionatore_bilanciato = WeightedRandomSampler(
    weights=pesi_campioni,
    num_samples=len(pesi_campioni),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=campionatore_bilanciato)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def mostra_immagini(immagini, etichette):
    fig, assi = plt.subplots(1, 8, figsize=(15, 3))
    for i in range(8):
        img = immagini[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        media = np.array([0.485, 0.456, 0.406])
        deviazione = np.array([0.229, 0.224, 0.225])
        img = deviazione * img + media 
        img = np.clip(img, 0, 1)
        
        assi[i].imshow(img)
        assi[i].set_title(nomi_classi[etichette[i]])
        assi[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Classi: {nomi_classi}")
    print(f"Dataset Bilanciato - Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    immagini_batch, etichette_batch = next(iter(train_loader)) 
    mostra_immagini(immagini_batch, etichette_batch)