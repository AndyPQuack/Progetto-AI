import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter

def get_data_loaders(data_dir, batch_size=256):
    train_transforms = transforms.Compose([
        transforms.CenterCrop(384),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(384),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_intero = datasets.ImageFolder(root=data_dir)
    etichette_reali = dataset_intero.targets

    indici_train, indici_test = train_test_split(
        list(range(len(dataset_intero))),
        test_size=0.20,
        stratify=etichette_reali,
        random_state=42
    )

    train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transforms), indici_train)
    test_dataset = Subset(datasets.ImageFolder(data_dir, transform=test_transforms), indici_test)

    etichette_train = [etichette_reali[i] for i in indici_train]
    conteggio_classi = Counter(etichette_train)
    pesi_classi = {classe: 1.0 / conteggio for classe, conteggio in conteggio_classi.items()}
    pesi_campioni = [pesi_classi[etichetta] for etichetta in etichette_train]
    pesi_campioni = torch.DoubleTensor(pesi_campioni)

    max_campioni = max(conteggio_classi.values())
    totale_campioni = max_campioni * 6 * 3

    campionatore_bilanciato = WeightedRandomSampler(
        weights=pesi_campioni,
        num_samples=totale_campioni,
        replacement=True
    )

    usa_gpu = torch.cuda.is_available()
    
    workers = 4 if usa_gpu else 2

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=campionatore_bilanciato, 
        num_workers=workers, 
        pin_memory=usa_gpu
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers, 
        pin_memory=usa_gpu
    )

    return train_loader, test_loader

def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto usando: {dispositivo}")

    train_loader, test_loader = get_data_loaders('dataset-resized', batch_size=32)

    modello = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_features = modello.fc.in_features
    modello.fc = nn.Linear(num_features, 6)
    modello = modello.to(dispositivo)

    criterio = nn.CrossEntropyLoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=0.0001)

    epoche = 10

    nome_file_csv = 'resnet.csv' 


    with open(nome_file_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoca', 'Loss_Train', 'Accuracy_Train'])

    for epoca in range(epoche):
        modello.train()
        loss_totale = 0.0
        corretti = 0
        totale = 0
        conteggio_pescate = torch.zeros(6, dtype=torch.int64).to(dispositivo)

        for immagini, etichette in train_loader:
            immagini, etichette = immagini.to(dispositivo), etichette.to(dispositivo)
            conteggio_pescate += torch.bincount(etichette, minlength=6)

            ottimizzatore.zero_grad()
            
            previsioni = modello(immagini)
            loss = criterio(previsioni, etichette)
            
            loss.backward()
            ottimizzatore.step()

            loss_totale += loss.item()
            _, previsti = torch.max(previsioni.data, 1)
            totale += etichette.size(0)
            corretti += (previsti == etichette).sum().item()

        accuratezza_train = 100 * corretti / totale
        print(f"\nDistribuzione classi pescate nell'epoca {epoca+1}: {conteggio_pescate.cpu().tolist()}")
        print(f"Epoca [{epoca+1}/{epoche}] | Loss: {loss_totale/len(train_loader):.4f} | Accuratezza Train: {accuratezza_train:.2f}%")
       
        with open(nome_file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoca + 1, round(loss_totale/len(train_loader), 4), round(accuratezza_train, 2)])
    print("\n--- INIZIO VALUTAZIONE SUL TEST SET ---")
    modello.eval()
    corretti_test = 0
    totale_test = 0
    
    with torch.no_grad():
        for immagini, etichette in test_loader:
            immagini, etichette = immagini.to(dispositivo), etichette.to(dispositivo)
            previsioni = modello(immagini)
            _, previsti = torch.max(previsioni.data, 1)
            totale_test += etichette.size(0)
            corretti_test += (previsti == etichette).sum().item()

    accuratezza_finale = 100 * corretti_test / totale_test
    print(f"Accuratezza Finale sul Test Set: {accuratezza_finale:.2f}%")

    nome_file_test = 'risultati_test.csv'
    file_esiste = os.path.isfile(nome_file_test)

    with open(nome_file_test, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_esiste:
            writer.writerow(['Modello', 'Accuratezza_Test'])
        
        writer.writerow(['resnet', round(accuratezza_finale, 2)])

    torch.save(modello.state_dict(), 'modello_resnet18_trashnet.pth')
    print("Modello ResNet18 salvato con successo!")

if __name__ == '__main__':
    main()