import os
import numba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, latent_dim, image_size=(64, 64), num_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_channels = num_channels  # Number of color channels (3 for RGB)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), latent_dim)
        self.fc_logvar = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (image_size[0] // 8) * (image_size[1] // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, image_size[0] // 8, image_size[1] // 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, latent_dim, image_size=(64, 64), num_channels=3):
        model = cls(latent_dim, image_size, num_channels)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model

class ImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(64, 64), cache_size=1000, max_files=1000000, shuf=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_files = os.listdir(root_dir)
        if not shuf: self.image_files.sort()
        self.image_files = self.image_files[:max_files]
        self.image_cache = {}  # Dictionary to store cached images
        self.cache_size = cache_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        if img_name in self.image_cache:
            image = self.image_cache[img_name]
        else:
            image = Image.open(img_name).convert('RGB')  # Load RGB image
            image = image.resize(self.image_size, Image.ANTIALIAS)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # Transpose to (C, H, W) format
            self.image_cache[img_name] = image
            while len(self.image_cache) >= self.cache_size:
                del self.image_cache[next(iter(self.image_cache))]
        return torch.tensor(image, dtype=torch.float32)

def train_vae(model, dataloader, epochs, device, lossfunc=1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    @numba.jit
    def custom_loss_function(recon_x, x, mu, logvar):
        if lossfunc == 1:
            return nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        elif lossfunc == 2:
            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return reconstruction_loss + kl_divergence
        elif lossfunc == 3:
            reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return reconstruction_loss + kl_divergence
        else:
            raise ValueError("Invalid loss function choice")

    epoch_tqdm = tqdm(range(epochs), desc=f'Epochs', unit='epoch')
    for epoch in epoch_tqdm:
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = custom_loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader.dataset)
        epoch_tqdm.set_description(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}')
    print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}', end='\r')

def generate_images(model, num_images, save_dir, device, debug=False):
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_images), desc='Generating Images', unit='image'):
            z = torch.randn(1, model.latent_dim).to(device)
            generated_image = model.decode(z).squeeze().cpu().numpy()
            pil_image = Image.fromarray((generated_image * 255).astype('uint8'), 'L')
            filename = f'generated_image_{i}.png'
            pil_image.save(os.path.join(save_dir, filename))
            if debug:
                print(f'Latent Space (z) for {filename}: {z.squeeze().cpu().numpy()}')

if __name__ == "__main__":
    latent_dim, batch_size, num_epochs = 10, 64, 5
    input_folder, model_save_path, output_folder = "in", "vae_model.pt", "out_generated"
    image_size = (64, 64)  # Change this to your desired image size
    
    os.makedirs(output_folder, exist_ok=True)
    dataset = ImageDataset(input_folder, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim, image_size, num_channels=3).to(device)
    
    train_vae(model, dataloader, num_epochs, device)
    model.save(model_save_path)
    generate_images(model, num_images=10, save_dir=output_folder, device=device)
