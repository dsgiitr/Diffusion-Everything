This process transforms data into noise through a series of steps, effectively allowing for the learning of a generative model that can later reconstruct data from noise. The forward diffusion process transforms the data distribution $p(x_0)$ (e.g., an image) into a series of increasingly noisy distributions through a Markov chain where 

$$
p(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

Or 

$$
x_t = \sqrt{1 - \beta_t} x_0 + \sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$  

This furthur can be reduced to 

$$
x_t = \sqrt{{\overline{\alpha}}_t} x_0 + \sqrt{1 - {\overline{\alpha}}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Where $\alpha_t = 1 - \beta_t$ and $\overline{\alpha}_t = \prod_{i=1}^{t} \alpha_i$  
Here is a simple code snippet for forward process : 
```python
def forward_process(self, images, timesteps, beta):
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, 0)
    shape = images.shape
    mean = alpha_cumprod[timesteps].sqrt().view(-1, *[1]*(len(shape)-1)) * images
    eps = (1 - alpha_cumprod[timesteps]).sqrt().view(-1, *[1]*(len(shape)-1))
    epsilon = torch.randn_like(images) 
    return mean + eps * epsilon, epsilon
```
