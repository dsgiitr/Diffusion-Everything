Code for DDPM Class Conditioned Sampler : 
```python
def generate(numImages, labels, steps, lerp = 1):
    #ns == Noise Scheduler
    alpha_cumprod = ns.alpha_cumprod[torch.linspace(0, timesteps - 1, steps).long()]
    alpha_cumprod_minus_one = torch.cat((torch.tensor([1]), alpha_cumprod[:-1]), dim = -1)
    beta = 1 - alpha_cumprod / alpha_cumprod_minus_one 
    alpha = 1 - beta
    beta_tilda = ((1 - alpha_cumprod_minus_one) / (1 - alpha_cumprod)) * beta 
    beta_final = beta * (1 - lerp) + beta_tilda * lerp
    sigma = beta_final.sqrt()
    idx = torch.linspace(0, timesteps - 1, steps).long()
    x_Ts = []
    x_T = torch.randn(numImages, *img_shape) 
    x_Ts.append(x_T)
    for t in torch.arange(steps - 1, -1, -1):
        epsilon_theta = UNet(x_T, idx[t], labels).sample 
        mean = (1 / alpha[t].sqrt()) * (x_T - ((1 - alpha[t])/(1 - alpha_cumprod[t]).sqrt()) * epsilon_theta) 
        old_x_T = x_T
        x_T = sample(mean, sigma[t])
        x_Ts.append(x_T)
    return x_Ts 
```