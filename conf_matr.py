import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def salva_matrice(modello, test_loader, classi, dispositivo, titolo, nome_file):
    modello.eval()
    vere, predette = [], []
    with torch.no_grad():
        for imm, etic in test_loader:
            prev = modello(imm.to(dispositivo))
            vere.extend(etic.numpy())
            predette.extend(torch.max(prev, 1)[1].cpu().numpy())
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(vere, predette), annot=True, fmt='d', cmap='Blues', xticklabels=classi, yticklabels=classi)
    plt.title(titolo, fontsize=16)
    plt.ylabel('Classe Reale')
    plt.xlabel('Classe Predetta')
    plt.tight_layout()
    plt.savefig(nome_file, dpi=300, bbox_inches='tight')
    plt.close()