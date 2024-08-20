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



#4. POKUSAJ ================================================================================================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Generator klasa sa Pixel Shuffle i većim kapacitetom
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),  # Koristi Pixel Shuffle umesto ConvTranspose2d
            nn.BatchNorm2d(1024 // 4),
            nn.ReLU(True),
            nn.Conv2d(1024 // 4, 512, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(512 // 4),
            nn.ReLU(True),
            nn.Conv2d(512 // 4, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(256 // 4),
            nn.ReLU(True),
            nn.Conv2d(256 // 4, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(128 // 4),
            nn.ReLU(True),
            nn.Conv2d(128 // 4, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator klasa sa Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.3, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.main(input)
        return self.sigmoid(out.view(-1))

# Parametri
nz = 100
lrG = 0.0002
lrD = 0.0001
beta1 = 0.5
epochs = 50
batch_size = 64

# Inicijalizacija modela
device = torch.device(args.device)
#device = torch.device("npu:0")
#torch.npu.set_device(device)
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

# Feature Matching Loss
def feature_matching_loss(real_features, fake_features):
    loss = torch.mean(torch.abs(real_features - fake_features))
    return loss

# Optimizeri
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

# Gradient Penalty funkcija
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    # Uskladite dimenzije
    fake_data_resized = nn.functional.interpolate(fake_data, size=real_data.size()[2:])
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data_resized)).requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


output_folder = "./outputMIAS3"  
os.makedirs(output_folder, exist_ok=True)

# Tensorboard za praćenje
writer = SummaryWriter()

# Listovi za praćenje gubitaka
losses_G = []
losses_D = []
# Logging
import logging
logging.basicConfig(filename='model4log.txt', level=logging.INFO, format='%(message)s')

from datetime import datetime

log_file = 'model4log.txt'

with open(log_file, 'w') as file:
    file.write(f'Log started on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

# Trening petlja
for epoch in range(epochs):
    for i, data in enumerate(data_loader, 0):
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        
        # Ažuriranje Discriminatora
        netD.zero_grad()
        output_real = netD(real_cpu)
        errD_real = -torch.mean(output_real)
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        errD_fake = torch.mean(output_fake)
        
        gradient_penalty = calc_gradient_penalty(netD, real_cpu.data, fake.data)
        errD = errD_real + errD_fake + 10 * gradient_penalty
        errD.backward()
        optimizerD.step()
        
        # Ažuriranje Generatora
        netG.zero_grad()
        output_fake = netD(fake)
        
        # Feature Matching Loss
        real_features = netD.main[:].forward(real_cpu).mean(dim=1)
        fake_features = netD.main[:].forward(fake).mean(dim=1)
        fm_loss = feature_matching_loss(real_features, fake_features)
        
        errG = -torch.mean(output_fake) + 0.1 * fm_loss
        errG.backward()
        optimizerG.step()

        # Čuvanje gubitaka
        losses_D.append(errD.item())
        losses_G.append(errG.item())
        
        # Štampanje rezultata
        if i % 100 == 0:
            logging.info(f'[{epoch}/{epochs}][{i}/{len(data_loader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

    # Čuvanje generisanih slika nakon svake epohe
    fake_resized = nn.functional.interpolate(fake, size=(64, 64))
    vutils.save_image(fake_resized.detach(), f'{output_folder}/fake_samples_epoch_{epoch}.png', normalize=True)

    # Čuvanje modela
    torch.save(netG.state_dict(), f'{output_folder}/netG_epoch_{epoch}.pth')
    torch.save(netD.state_dict(), f'{output_folder}/netD_epoch_{epoch}.pth')


# Zatvaranje writer-a
writer.close()

