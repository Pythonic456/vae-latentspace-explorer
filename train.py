import os, time, math, random
import torch
from main import VAE, ImageDataset, train_vae
from PIL import Image, ImageDraw, ImageTk
import numpy as np

# Parameters
pow = 9
latent_dim = 30
batch_size = 200
num_epochs = 10000
image_size = (2**pow, 2**pow)
lossfunc = 3
cache_size = 5000
max_files = 100
input_folder = 'in/'  # "tmp/dst/"
model_save_path = "vae_model.pt"

# Train VAE
dataset = ImageDataset(input_folder, image_size, cache_size, max_files)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim, image_size, num_channels=3).to(device)  # Specify num_channels as 3 for RGB images
train_vae(model, dataloader, num_epochs, device, lossfunc=lossfunc)
model.save(model_save_path)
