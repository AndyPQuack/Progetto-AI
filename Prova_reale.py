import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from CNN import TrashNetCNN
from CNN_emp import TrashNetCNN_emp

def main():
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n---> Avvio Test Reale ('Inference in the wild') su: {disp} <--- \n")

    classi = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    trasformazione = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Caricamento dei pesi neurali in corso...")
    
    modello_base = TrashNetCNN().to(disp)
    modello_base.load_state_dict(torch.load('modello_trashnet.pth', map_location=disp))
    modello_base.eval()

    modello_emp = TrashNetCNN_emp().to(disp)
    modello_emp.load_state_dict(torch.load('modello_trashnet_emp.pth', map_location=disp))
    modello_emp.eval()

    modello_res = models.resnet18(weights=None)
    modello_res.fc = nn.Linear(modello_res.fc.in_features, 6)
    modello_res = modello_res.to(disp)
    modello_res.load_state_dict(torch.load('modello_resnet.pth', map_location=disp))
    modello_res.eval()

    cartella_foto = 'foto_jpeg'
    file_csv_output = 'risultati_reali.csv'
    
    if not os.path.exists(cartella_foto):
        print(f"ERRORE: La cartella '{cartella_foto}' non esiste.")
        return

    elenco_file = [f for f in os.listdir(cartella_foto) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(elenco_file) == 0:
        print(f"ERRORE: Nessuna immagine in '{cartella_foto}'.")
        return

    print("\n" + "="*60)
    print(" INIZIO VALUTAZIONE FOTO REALI ")
    print("="*60 + "\n")

    with open(file_csv_output, mode='w', newline='', encoding='utf-8') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(['Immagine', 'Modello', 'Predizione', 'Confidenza'])

        with torch.no_grad():
            for nome_file in elenco_file:
                percorso = os.path.join(cartella_foto, nome_file)
                try:
                    img_pil = Image.open(percorso).convert('RGB')
                    img_tensore = trasformazione(img_pil).unsqueeze(0).to(disp) 
                    
                    out_base = modello_base(img_tensore)
                    out_emp = modello_emp(img_tensore)
                    out_res = modello_res(img_tensore)
                    
                    prob_base = F.softmax(out_base, dim=1).squeeze()
                    prob_emp = F.softmax(out_emp, dim=1).squeeze()
                    prob_res = F.softmax(out_res, dim=1).squeeze()
                    
                    idx_base = torch.argmax(prob_base).item()
                    idx_emp = torch.argmax(prob_emp).item()
                    idx_res = torch.argmax(prob_res).item()

                    pred_base = classi[idx_base].upper()
                    pred_emp = classi[idx_emp].upper()
                    pred_res = classi[idx_res].upper()
                    
                    conf_base = f"{prob_base[idx_base]*100:.1f}%"
                    conf_emp = f"{prob_emp[idx_emp]*100:.1f}%"
                    conf_res = f"{prob_res[idx_res]*100:.1f}%"

                    writer.writerow([nome_file, 'CNN Base', pred_base, conf_base])
                    writer.writerow([nome_file, 'CNN EMP', pred_emp, conf_emp])
                    writer.writerow([nome_file, 'ResNet18', pred_res, conf_res])

                    print(f"Elaborato: {nome_file}")
                    
                except Exception as e:
                    print(f"Errore su {nome_file}: {e}")

    print(f"\n---> Finito! Ho creato il file: {file_csv_output} <---")

if __name__ == '__main__':
    main()