import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import noise_scheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import plotly.graph_objects as go
import torchvision.transforms as T
from scipy.stats import gaussian_kde

dataset_path = os.environ["DATASET"]

def generateFigure(data, labels):
    unique_labels = np.unique(labels)
    xmin, ymin = data.min(axis=0) - 1
    xmax, ymax = data.max(axis=0) + 1
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, label in enumerate(unique_labels):
        class_data = data[labels == label].T
        kde = gaussian_kde(class_data)
        z = np.reshape(kde(positions).T, x.shape)
        fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False, name=f'Class {label}'))

    fig.update_layout(title='3D Kernel Density Estimation on Latent Space',
                      scene=dict(
                          xaxis_title='X Axis',
                          yaxis_title='Y Axis',
                          zaxis_title='Density',
                          aspectratio=dict(x=1, y=1, z=0.5)
                      ))
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
    )
    return fig


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, in_channels), 
            nn.Tanh(), 
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), 
            nn.GroupNorm(4, out_channels), 
            nn.Tanh(), 
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1), 
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0) 
    def forward(self, x):
        residue = x 
        x = self.net(x)
        x = x + self.residual(residue)
        return x 


class VAE(nn.Module):
    def __init__(self, device = "cpu"): 
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1), 
            ResidualBlock(32, 64), 
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 0), 
            ResidualBlock(64, 64), 
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 0), 
            nn.Flatten(), 
            nn.Linear(4608//2, 4), 
        )
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self.l1 = nn.Linear(2, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), 
            ResidualBlock(64, 64), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), 
            ResidualBlock(64, 32), 
            nn.Conv2d(32, 1, kernel_size = 3, padding = 1), 
            nn.Tanh(), 
        )
    def forward(self, x):
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim = -1)
        sample = torch.randn_like(mean) * log_var.exp() * 1.5 + mean
        x = self.l1(sample)
        x = x.view(-1, 64, 7, 7) 
        x = self.decoder(x)
        return x, mean, log_var

    @torch.inference_mode()
    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim = -1)
        return mean, log_var

    @torch.inference_mode()
    def decode(self, mean):
        x = self.l1(mean)
        x = x.view(-1, 64, 7, 7) 
        x = self.decoder(x)
        return x  

    @torch.inference_mode()
    def get_random_encoding(self, num_images):
        items = [self.dataset[i] for i in range(num_images)]
        x = [i[0] for i in items]
        y = [i[1] for i in items]
        x = torch.stack(x).to(self.device)
        return x, y



    @torch.inference_mode()
    def get_random_encoding(self, num_images):
        items = [self.dataset[i] for i in range(num_images)]
        x = [i[0] for i in items]
        y = [i[1] for i in items]
        x = torch.stack(x).to(self.device)
        return x, y
    

    def train(self, num_epochs, batch_size, num_test, streamlit_callback = None, progress = None):
        self.dataset = torchvision.datasets.MNIST(root = dataset_path, download = True, transform = self.preprocess)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size,  num_workers = 4)
        optimizer = torch.optim.Adam(self.parameters())
        it = 0 
        testx, testy = self.get_random_encoding(num_test)
        for epoch in range(num_epochs):
            for i, (x, y) in tqdm(enumerate(dataloader), total = len(dataloader)):
                x = x.to(self.device)
                recon, mean, log_var = self(x)
                recon_loss = F.mse_loss(recon, x, reduction = 'sum')
                kl_loss = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) #KL Div
                loss = recon_loss + kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                test_means = self.encode(testx)[0].cpu()
                if streamlit_callback is not None :
                    streamlit_callback(test_means, testy)
                if progress is not None :         
                    progress.progress(it / (num_epochs * len(dataloader)), text = f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(dataloader)}]")
                it += 1 


class DiffusionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, timesteps):
        super().__init__()
        self.layer = nn.Linear(in_dim + 10, out_dim)
        self.ts_embed = nn.Embedding(timesteps, out_dim)
    def forward(self, x, labels, ts):
        x = torch.cat([x, F.one_hot(labels, num_classes = 10)], dim = -1)
        x = self.layer(x)
        x = F.relu(x)
        x = x + self.ts_embed(ts)
        return x

class UNet1D(nn.Module):
    def __init__(self, timesteps,  device = 'cpu'):
        super().__init__()
        self.net = nn.ModuleList([
            DiffusionMLP(2, 128, timesteps), 
            DiffusionMLP(128, 256, timesteps), 
            DiffusionMLP(256, 128, timesteps), 
            DiffusionMLP(128, 2, timesteps), 
        ])
        self.device = device
    def forward(self, x, labels, ts):
        for module in self.net :
            x = module(x, labels, ts) 
        return x
    

class Diffusion():
    def __init__(self, unet, vae, device, scheduler, timesteps, beta_start = 1e-3, beta_end = 1e-2):
        if scheduler == "Cosine":
            self.ns = noise_scheduler.Cosine(timesteps, device = device)
        else :
            self.ns = noise_scheduler.Linear(
                    beta_start = beta_start, 
                    beta_end = beta_end, 
                    timesteps = timesteps
            )
        self.timesteps = timesteps
        self.unet = unet 
        self.vae = vae  
        self.device = device
        means = [] 
        labels = []

        dataset = torchvision.datasets.MNIST(root = dataset_path, download = True, transform = vae.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2048, drop_last = True, num_workers = 3)
        for (x, y) in tqdm(dataloader, total = len(dataloader)) :
            x = x.to(device)
            with torch.inference_mode() :
                means.append(vae.encode(x)[0])
                labels.append(y)
        self.labels = torch.cat(labels, dim = 0).cpu()
        self.all_means = torch.cat(means, dim = 0).cpu()
        self.m = self.all_means.mean(dim = 0)
        self.s = self.all_means.std(dim = 0)
        self.all_means = (self.all_means - self.m) / (self.s) #Standardize

        self.m = self.m.to(device)
        self.s = self.s.to(device)


    def renorm(self, img):
        return self.s * img + self.m 
    
    def tensor2numpy(self, img):
        return ((img + 1)/2).clamp(0, 1)[:, 0, :, :].cpu().numpy()
            
    
    @torch.inference_mode()
    def generate(self, label, streamlit_callback):
        x_T = torch.randn(1, 2, device = self.device) 
        ns = self.ns
        for t in tqdm(torch.arange(self.timesteps - 1, -1, -1, device = self.device)):
            z = torch.randn_like(x_T)
            epsilon_theta = self.unet(x_T, torch.LongTensor([label]).to(self.device), torch.LongTensor([t]).to(self.device))
            mean = (1 / ns.alpha[t].sqrt()) * (x_T - ((1 - ns.alpha[t])/(1 - ns.alpha_cumprod[t]).sqrt()) * epsilon_theta) 
            x_T = mean + z * ns.sigma[t]
            data = self.vae.decode(self.renorm(x_T))
            img = self.tensor2numpy(data)[0]
            if streamlit_callback is not None :
                streamlit_callback(t, img, (self.renorm(x_T)).detach().cpu().tolist()[0])

    def train(self, num_iters, streamlit_callback = None):
        ns = self.ns 
        optimizer = torch.optim.Adam(self.unet.parameters(), lr = 1e-3)
        for epoch in tqdm(range(num_iters)):
            data, label = self.get_batch(128)
            data = data.to(self.device)
            label = label.to(self.device)
            ts = torch.randint(0, self.timesteps, (128, ), device = self.device)
            data, eps = ns.forward_process(data, ts)
            noise = self.unet(data, label, ts)
            loss = F.mse_loss(noise, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if streamlit_callback is not None :
                streamlit_callback(epoch, loss.item())
            
    def get_batch(self, batch_size):
        idx = torch.randint(0, len(self.all_means), (batch_size, ))
        return self.all_means[idx], self.labels[idx]

