import torch
import torchvision.transforms as T

class ReverseSampler():
    def __init__(self, timesteps, noise_scheduler, unet, device, img_shape):
        self.timesteps = timesteps
        self.noise_scheduler = noise_scheduler 
        self.device = device 
        self.UNet = unet
        self.img_shape = img_shape

        self.renorm = T.Compose([
            T.Normalize([-1] * img_shape[0], [2] * img_shape[0]),
        ])
    def reverse(self):
        pass

    @staticmethod
    def sample(mean, std):
        return mean + std * torch.randn_like(mean)
    
    def tensor2numpy(self, images):
        images = self.renorm(images.detach().cpu())
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images  
