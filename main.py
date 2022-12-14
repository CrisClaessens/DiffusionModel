import torch
import torchvision

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from unet import SimpleUnet
from trainer import Trainer
from losses import L1

# Define image and batch size
IMG_SIZE = 64
BATCH_SIZE = 128

# Compose image transforms
data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
data_transform = transforms.Compose(data_transforms)

# Generate dataloaders
train_data = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform)
val_data = torchvision.datasets.StanfordCars(root=".", download=True, split='test', transform=data_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Set device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Load U-Net model
model = SimpleUnet().to(device)

# Set training settings
loss = F.l1_loss

optimizer = Adam(model.parameters(), lr=0.001)

# Load trainer
trainer = Trainer(model=model, device=device, criterion=loss, optimizer=optimizer, training_dataLoader=train_loader,
                  validation_dataLoader=val_loader, T=200, epochs=100)

training_losses, validation_losses, lr_rates = trainer.run_trainer()

