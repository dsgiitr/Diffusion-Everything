Enjoying ? Ok now we'll be training a conditional diffusion model on 2D latent space which will help us conditionally sample points from the latent space and then decode the point from the VAE decoder which was trained above. 
We used a conditional UNet which takes 2D input and transforms it accordingly `2 -> 128 -> 256 -> 128 -> 2`. Conditioning is done by concatenating with one-hot labels before every transformation. Here is the code for Conditional UNet : 
```python
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
```
For reverse process (denoising) we have used the DDPM solver 
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_{\theta}(x_t, t)) - \sigma_t z
$$
Here is the code for reverse process : 
```python
x_T = torch.randn(1, 2) 
ns = self.ns
for t in torch.arange(timesteps - 1, -1, -1):
    z = torch.randn_like(x_T)
    epsilon_theta = self.unet(x_T, torch.LongTensor([label], torch.LongTensor([t]))
    mean = (1 / ns.alpha[t].sqrt()) * (x_T - ((1 - ns.alpha[t])/(1 - ns.alpha_cumprod[t]).sqrt()) * epsilon_theta) 
    x_T = mean + z * ns.sigma[t]
    image = self.vae.decode(self.renorm(x_T))
```