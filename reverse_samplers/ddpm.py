import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
from tqdm.auto import tqdm
from sampler import ReverseSampler


class DDPMSampler(ReverseSampler):
    @torch.inference_mode
    def reverse(self, numImages, labels, steps, lerp = 1, streamlit_callback = None):
        assert(numImages == len(labels))
        ns = self.noise_scheduler
        alpha_cumprod = ns.alpha_cumprod[torch.linspace(0, self.timesteps - 1, steps, device = self.device).long()]
        alpha_cumprod_minus_one = torch.cat((torch.tensor([1], device = self.device), alpha_cumprod[:-1]), dim = -1)
        beta = 1 - alpha_cumprod / alpha_cumprod_minus_one 
        alpha = 1 - beta
        beta_tilda = ((1 - alpha_cumprod_minus_one) / (1 - alpha_cumprod)) * beta 
        beta_final = beta * (1 - lerp) + beta_tilda * lerp
        sigma = beta_final.sqrt()
        idx = torch.linspace(0, self.timesteps - 1, steps, device = self.device).long()
        x_Ts = []
        x_T = torch.randn(numImages, *self.img_shape, device = self.device) 
        x_Ts.append(self.tensor2numpy(x_T))
        for t in tqdm(torch.arange(steps - 1, -1, -1, device = self.device)):
            epsilon_theta = self.UNet(x_T, idx[t], labels).sample 
            mean = (1 / alpha[t].sqrt()) * (x_T - ((1 - alpha[t])/(1 - alpha_cumprod[t]).sqrt()) * epsilon_theta) 
            old_x_T = x_T
            x_T = self.sample(mean, sigma[t])
            if streamlit_callback :
                streamlit_callback(epsilon_theta, mean, old_x_T, self.tensor2numpy(x_T.cpu()), t)
            x_Ts.append(self.tensor2numpy(x_T.cpu()))
        return x_Ts 