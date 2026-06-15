import torch, csv, conf_matr
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
from CNN import TrashNetCNN

def get_data_loaders(data_dir, batch_size=32):
    t_tr = transforms.Compose([transforms.CenterCrop(384), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_te = transforms.Compose([transforms.CenterCrop(384), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(root=data_dir)
    idx_tr, idx_te = train_test_split(list(range(len(ds))), test_size=0.20, stratify=ds.targets, random_state=128)
    ds_tr, ds_te = Subset(datasets.ImageFolder(data_dir, transform=t_tr), idx_tr), Subset(datasets.ImageFolder(data_dir, transform=t_te), idx_te)
    et_tr = [ds.targets[i] for i in idx_tr]
    cc = Counter(et_tr)
    pesi = torch.DoubleTensor([1.0 / cc[e] for e in et_tr])
    samp = WeightedRandomSampler(pesi, max(cc.values()) * 6, replacement=True)
    gpu = torch.cuda.is_available()
    w = 4 if gpu else 2
    return DataLoader(ds_tr, batch_size=batch_size, sampler=samp, num_workers=w, pin_memory=gpu), DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=w, pin_memory=gpu), ds.classes

def main():
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_l, te_l, classi = get_data_loaders('dataset-resized', batch_size=32)
    modello = TrashNetCNN().to(disp)
    criterio = nn.CrossEntropyLoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)
    epoche, record_loss = 15, float('inf')
    
    with open('CNN.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['Epoca', 'Loss_Train', 'Accuracy_Train', 'Loss_Test', 'Accuracy_Test'])

    for ep in range(epoche):
        modello.train()
        loss_tr, corr_tr, tot_tr = 0.0, 0, 0
        for imm, etic in tr_l:
            imm, etic = imm.to(disp), etic.to(disp)
            ottimizzatore.zero_grad()
            prev = modello(imm)
            loss = criterio(prev, etic)
            loss.backward()
            ottimizzatore.step()
            loss_tr += loss.item()
            corr_tr += (torch.max(prev.data, 1)[1] == etic).sum().item()
            tot_tr += etic.size(0)

        modello.eval()
        loss_te, corr_te, tot_te = 0.0, 0, 0
        with torch.no_grad():
            for imm, etic in te_l:
                imm, etic = imm.to(disp), etic.to(disp)
                prev = modello(imm)
                loss_te += criterio(prev, etic).item()
                corr_te += (torch.max(prev.data, 1)[1] == etic).sum().item()
                tot_te += etic.size(0)

        acc_tr, acc_te = 100 * corr_tr / tot_tr, 100 * corr_te / tot_te
        m_loss_tr, m_loss_te = loss_tr / len(tr_l), loss_te / len(te_l)
        print(f"Ep [{ep+1}/{epoche}] | TrL: {m_loss_tr:.4f} | TrA: {acc_tr:.2f}% | TeL: {m_loss_te:.4f} | TeA: {acc_te:.2f}%")

        with open('CNN.csv', 'a', newline='') as f:
            csv.writer(f).writerow([ep + 1, round(m_loss_tr, 4), round(acc_tr, 2), round(m_loss_te, 4), round(acc_te, 2)])


        if m_loss_te < record_loss:
            record_loss = m_loss_te
            print(f"  -> Nuovo record! Loss minima raggiunta: {record_loss:.4f}. Modello salvato.")
            torch.save(modello.state_dict(), 'modello_trashnet_emp.pth')

    modello.load_state_dict(torch.load('modello_trashnet.pth', map_location=disp))
    conf_matr.salva_matrice(modello, te_l, classi, disp, 'Matrice di Confusione - CNN Custom', 'matrice_confusione_cnn.png')

if __name__ == '__main__':
    main()