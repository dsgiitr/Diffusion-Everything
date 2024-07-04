import torch

class NoiseScheduler():
    def __init__(self):
        self.alpha = None
        self.beta = None 
        self.alpha_cumprod = None 
        self.alpha_cumprod_minus_one = None 
        self.sigma = None 
    def forward_process(self, images, timesteps):
        shape = images.shape

        mean = self.alpha_cumprod[timesteps].sqrt().view(-1, *[1]*(len(shape)-1)) * images
        eps = (1 - self.alpha_cumprod[timesteps]).sqrt().view(-1, *[1]*(len(shape)-1))

        epsilon = torch.randn_like(images) 

        return mean + eps * epsilon, epsilon
    
class Linear(NoiseScheduler):
    def __init__(self, 
        beta_start = 1e-4, 
        beta_end = 2e-2, 
        timesteps = 1000, 
        device = None
    ): 
        if device == None :
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else :
            self.device = device

        self.beta_start = beta_start 
        self.beta_end = beta_end
        self.timesteps = timesteps 

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device = self.device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, 0)
        self.alpha_cumprod_minus_one = torch.cat((torch.tensor([1], device = self.device), self.alpha_cumprod[:-1]), dim = -1)
        self.sigma = self.beta.sqrt()
    

class Cosine(NoiseScheduler):
    def __init__(self, 
        timesteps = 1000, 
        device = None
    ):

        if device == None :
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else :
            self.device = device
        
        t = torch.arange(1, timesteps+1)
        t_max = timesteps
        self.alpha_cumprod = (self.f(t, t_max) / self.f(torch.tensor(0), t_max)).to(self.device)
        self.alpha_cumprod_minus_one = torch.cat((torch.tensor([1], device = self.device), self.alpha_cumprod[:-1]), dim = -1)
        self.alpha = self.alpha_cumprod / self.alpha_cumprod_minus_one 
        self.beta = (1 - self.alpha).clamp(0, 0.999)
        self.sigma = self.beta.sqrt() 

    @staticmethod
    def f(t, T, s = 8e-3):
        return torch.cos(((t/T+s) / (1+s)) * (torch.pi/2)) ** 2 


        
