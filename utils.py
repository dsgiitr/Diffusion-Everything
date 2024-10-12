import torch
import torchvision.transforms as T

renorm = T.Compose([
            T.Normalize([-1, -1, -1] , [2, 2, 2]),
        ])


preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)
        ])

def tensor2numpy(images):
        images = renorm(images.detach().cpu())
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images 