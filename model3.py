import argparse

parser = argparse.ArgumentParser(description='Set PyTorch device.')
parser.add_argument('device', type=str, choices=['cpu', 'npu:0'], help='Device to use for PyTorch operations')
args = parser.parse_args()



#MIAS pytorch data loader

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Putanja do MIAS dataset-a i info.txt datoteke
folder_path = '/miep/mammo_files_v2/all-mias'
info_file = '/miep/mammo_files_v2/Info.txt'

# Funkcija za učitavanje informacija iz info.txt
def load_info(info_file):
    with open(info_file, 'r') as f:
        lines = f.readlines()

    info = []
    for line in lines:
        line = line.strip().split()
        if len(line) >= 7 and 'mdb' in line[0]:  # Treba nam minimalno 7 kolona za ROI (x, y, radius)
            ref_number = line[0]
            # Filtriramo slike koje postoje od mdb001 do mdb322
            if 'mdb' in ref_number and 1 <= int(ref_number[3:6]) <= 322:
                data = {
                    'reference_number': ref_number,
                    'background_tissue': line[1],
                    'abnormality_class': line[2],
                    'abnormality_severity': line[3],
                    'x_coordinate': int(line[4]) if line[4].isdigit() else None,
                    'y_coordinate': int(line[5]) if line[5].isdigit() else None,
                    'radius': int(line[6]) if line[6].isdigit() else None
                }
                info.append(data)

    return info

# Klasa Dataset za učitavanje slika i njihovih ROI-ova
class MIASDataset(Dataset):
    def __init__(self, folder_path, info_file, transform=None):
        self.folder_path = folder_path
        self.info = load_info(info_file)
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        entry = self.info[idx]
        ref_number = entry['reference_number']
        x = entry['x_coordinate']
        y = entry['y_coordinate']
        radius = entry['radius']
        abnormality_class = entry['abnormality_class']

        image_path = os.path.join(self.folder_path, f"{ref_number}.pgm")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} could not be read.")

        if x is None or y is None or radius is None or abnormality_class == 'NORM':
            # Handle cases where there is no ROI or the image is normal
            roi = cv2.resize(image, (299, 299))
        else:
            # Extract ROI if coordinates and radius are valid
            roi = self.extract_roi(image, x, y, radius)

        if roi is None:
            roi = np.zeros((299, 299), dtype=np.uint8)

        if self.transform:
            roi = self.transform(roi)

        return roi, abnormality_class

    def extract_roi(self, image, x, y, radius):
        start_x = max(x - radius, 0)
        start_y = max(y - radius, 0)
        end_x = min(x + radius, image.shape[1])
        end_y = min(y + radius, image.shape[0])

        roi = image[start_y:end_y, start_x:end_x]
        return roi

# Primer transformacije za normalizaciju i promenu veličine ROI-a
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if __name__ == "__main__":
    # Kreiranje instance Dataset-a
    dataset = MIASDataset(folder_path, info_file, transform=transform)

    # Kreiranje DataLoader-a
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Prikaz nekih ROI-ova za pregled iz DataLoader-a
    for batch_idx, (rois, labels) in enumerate(data_loader):
        if batch_idx >= 1:  # Prikazi samo prvu seriju
            break

        plt.figure(figsize=(20, 20))
        for i in range(len(rois)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(rois[i].squeeze(), cmap='gray')  # squeeze za skidanje dodatne dimenzije
            plt.title(f'Class: {labels[i]}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

#CBIS-DDSM pytorch data loader

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CBISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self._get_file_list()
   
    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_list.append(os.path.join(root, file))
        return file_list
   
    def __len__(self):
        return len(self.file_list)
   
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        np_image = np.load(self.file_list[idx])
       
        # Convert numpy array to torch tensor with a specific dtype
        np_image = torch.tensor(np_image, dtype=torch.float32)
       
        # Extract label from filename
        filename = os.path.basename(self.file_list[idx])
        label_str = filename.split('_')[0].split('-')[0]
        label = int(label_str)
       
        if self.transform:
            np_image = self.transform(np_image)
       
        return np_image, label

# Example usage:
dataset = CBISDataset(root_dir='/miep/mammo_files_v2/Datasets in npy format/CBIS-DDSMnpy')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterating through the dataloader:
for images, labels in dataloader:
    # Your training/validation loop here
    pass

#DDSM pytorch data loader

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class DDSMDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # Pristupanje svim folderima i fajlovima u root_dir
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".npy"):
                    self.image_paths.append(os.path.join(root, file))
                    # Određivanje labela na osnovu naziva fajla
                    if "_mask1.npy" in file or "_mask2.npy" in file:
                        self.labels.append(1)  # Sadrži abnormalnosti
                    else:
                        self.labels.append(0)  # Ne sadrži abnormalnosti
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        label = self.labels[idx]
        
        # Konverzija u tensor i dodavanje dimenzije kanala za grayscale slike
        image = torch.FloatTensor(image).unsqueeze(0)
        label = torch.LongTensor([label])
        
        return image, label

# Kreiranje DataLoader-a za svaki tip (Normal, Benign, Cancer)
root_dirs = [
    "/miep/mammo_files_v2/Datasets in npy format/DDSMnpy/Normal",
    "/miep/mammo_files_v2/Datasets in npy format/DDSMnpy/Benign",
    "/miep/mammo_files_v2/Datasets in npy format/DDSMnpy/Cancer"
]

dataloaders = []
for root_dir in root_dirs:
    dataset = DDSMDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloaders.append(dataloader)

# Primer upotrebe DataLoader-a za iteriranje kroz podatke
for dataloader in dataloaders:
    for batch in dataloader:
        images, labels = batch
        # Obrada batch-a slika i labela

###################################################################################################################################
#3. GRUPA
###################################################################################################################################

import torch

# Funkcija za ispisivanje tipova i oblika batch-a iz DataLoader-a
def inspect_dataloader(dataloader):
    for i, batch in enumerate(dataloader):
        images, labels = batch
        print(f"Batch {i+1}")
        print(f"Images type: {type(images)}")
        print(f"Images shape: {images.shape}")
        
        # Provera tipa i strukture labele
        print(f"Labels type: {type(labels)}")
        if isinstance(labels, torch.Tensor):
            print(f"Labels shape: {labels.shape}")
        elif isinstance(labels, tuple):
            print(f"Labels length: {len(labels)}")
            for j, label in enumerate(labels):
                print(f"Label {j} type: {type(label)}, shape: {label.shape if isinstance(label, torch.Tensor) else 'N/A'}")
        else:
            print(f"Labels content: {labels}")
        break  # Zaustavljamo se nakon prvog batch-a

# Provera za MIAS DataLoader
print("MIAS DataLoader:")
inspect_dataloader(data_loader)

# Provera za CBIS-DDSM DataLoader
print("\nCBIS-DDSM DataLoader:")
inspect_dataloader(dataloader)

# Provera za DDSM DataLoader-e
for idx, dataloader in enumerate(dataloaders):
    print(f"\nDDSM DataLoader {idx+1}:")
    inspect_dataloader(dataloader)

import torch_npu
print(torch.npu.is_available())  # True
print(torch.npu.device_count())  # 2
print(torch.npu.current_device())



#3. POKUSAJ ================================================================================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# Generator klasa sa većim kapacitetom
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(1, 1, kernel_size=5, padding=2)  # Dodavanje konvolucionog sloja
        )

    def forward(self, input):
        return self.main(input)


# Discriminator klasa sa LeakyReLU sa većom negativnom nagibom
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),  # Povećana negativna nagiba
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),  # Povećana negativna nagiba
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3, inplace=True),  # Povećana negativna nagiba
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3, inplace=True),  # Povećana negativna nagiba
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.main(input)
        return self.sigmoid(out.view(-1))

# Parametri
nz = 100  # Veličina latentnog vektora
lrG = 0.0002
lrD = 0.0001  # Smanjena brzina učenja za diskriminator
beta1 = 0.5
epochs = 50
batch_size = 64  # Povećan batch size

# Inicijalizacija modela
device = torch.device(args.device)
#device = torch.device("npu:0")
#torch.npu.set_device(device)
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

# WGAN-GP Loss funkcija
def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)

    interpolated_output = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

lambda_gp = 10  # Regularizacija za gradijentnu penalizaciju

# Optimizeri
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

# Tensor za bucne vektore
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Inicijalizujte writer i folder za izlaz
writer = SummaryWriter()
output_folder = "./outputMIAS2"  
os.makedirs(output_folder, exist_ok=True)

# Inicijalizujte liste za čuvanje gubitaka
losses_D = []
losses_G = []

def convert_labels_to_tensor(labels):
    label_dict = {'MISC': 0, 'CALC': 1, 'ASYM': 2, 'ARCH': 3, 'CIRC': 4, 'SPIC': 5}
    numeric_labels = [label_dict[label] for label in labels]
    return torch.tensor(numeric_labels, dtype=torch.float, device=device)
   
# Logging
import logging
logging.basicConfig(filename='model3log.txt', level=logging.INFO, format='%(message)s')

from datetime import datetime

log_file = 'model3log.txt'

with open(log_file, 'w') as file:
    file.write(f'Log started on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

# Treniranje modela
for epoch in range(epochs):
    for i, data in enumerate(data_loader, 0):  # MIAS dataloader
        # (1) Ažuriranje D mreže: log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        
        # Konvertovanje labela u tensor ako su stringovi
        if isinstance(data[1], tuple):
            labels = convert_labels_to_tensor(data[1]).to(device)
        else:
            labels = data[1].to(device)
        
        # Forward prolaz kroz diskriminator za stvarne slike
        output_real = netD(real_cpu).view(batch_size, -1).mean(dim=1)
        errD_real = -torch.mean(output_real)  # Promena za WGAN-GP

        # Generisanje lažnih slika i forward prolaz kroz diskriminator
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_resized = F.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)  # Promena veličine generisanih slika
        output_fake = netD(fake_resized.detach()).view(batch_size, -1).mean(dim=1)
        errD_fake = torch.mean(output_fake)  # Promena za WGAN-GP

        # Gradijentna penalizacija
        real_cpu_resized = F.interpolate(real_cpu, size=(299, 299), mode='bilinear', align_corners=False)  # Promena veličine stvarnih slika
        gp = gradient_penalty(netD, real_cpu_resized.data, fake_resized.data)
        errD = errD_real + errD_fake + lambda_gp * gp

        errD.backward()
        optimizerD.step()

        # (2) Ažuriranje G mreže: log(D(G(z)))
        netG.zero_grad()
        output_fake = netD(fake_resized).view(batch_size, -1).mean(dim=1)
        errG = -torch.mean(output_fake)  # Promena za WGAN-GP
        errG.backward()
        optimizerG.step()

        # Beleženje gubitaka
        losses_D.append(errD.item())
        losses_G.append(errG.item())

        logging.info(f'[{epoch}/{epochs}][{i}/{len(data_loader)}] '
              f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
              f'D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f}')

        if i % 100 == 0:
            vutils.save_image(real_cpu, f'{output_folder}/real_samples_epoch_{epoch}.png', normalize=True)
            fake = netG(fixed_noise)
            fake_resized = F.interpolate(fake, size=(299, 299), mode='bilinear', align_corners=False)
            vutils.save_image(fake_resized.detach(), f'{output_folder}/fake_samples_epoch_{epoch}.png', normalize=True)

    # Sačuvajte model posle svake epohe
    torch.save(netG.state_dict(), f'{output_folder}/netG_epoch_{epoch}.pth')
    torch.save(netD.state_dict(), f'{output_folder}/netD_epoch_{epoch}.pth')

# Zatvaranje writer-a
writer.close()


