import torch
import torch.nn as nn
import torch.utils
import torchvision
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import warnings
from torchvision.datasets import ImageFolder
from diffusers import UNet2DModel, UNet2DConditionModel
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
warnings.filterwarnings("ignore")



preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

renorm = T.Compose([
    T.Normalize([-1], [2]),
])

class DDIM():
    def __init__(self, betaStart, betaEnd,
                 timesteps, UNetConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.betaStart = betaStart
        self.betaEnd = betaEnd 
        self.timesteps = timesteps
        self.UNet = UNet2DModel(**UNetConfig).to(self.device)

        #DDIM Hyperparameters
        self.betas = torch.linspace(betaStart, betaEnd, timesteps, device = self.device)
        self.alphas = 1 - self. betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.sigmas = self.betas.sqrt()

    def train(self, dataloader, numEpochs, checkpoint,
              logStep = 100, checkpointStep = 195, lr = 1e-3):
        optimizer = torch.optim.Adam(self.UNet.parameters(), lr = lr)
        for epoch in range(numEpochs):
            print(f"Epoch [{epoch+1}/{numEpochs}]")
            for i, (batch, y) in tqdm(enumerate(dataloader), total = len(dataloader)):

                batch = batch.to(self.device)
                ts = torch.randint(0, self.timesteps, (batch.shape[0], ))
                encodedImages, epsilon = self.addNoise(batch, ts)
                predictedNoise = self.UNet(encodedImages, ts).sample
                loss = F.mse_loss(predictedNoise, epsilon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % logStep == 0 :
                    tqdm.write(f"Step : {i+1} | Loss : {loss.item()}")
                if (i+1) & checkpointStep == 0 :
                    torch.save({
                        "model" : self.UNet.state_dict(), 
                        "beta_start" : self.betaStart,
                        "beta_end" : self.betaEnd,
                        "timesteps" : self.timesteps
                    }, checkpoint)
            torch.save({
                        "model" : self.UNet.state_dict(), 
                        "beta_start" : self.betaStart,
                        "beta_end" : self.betaEnd,
                        "timesteps" : self.timesteps
                    }, checkpoint)
    
    def addNoise(self, images, timesteps):
        mean = self.alpha_cumprod.sqrt()[timesteps].view(-1, 1, 1, 1) * images
        std = (1 - self.alpha_cumprod).sqrt()[timesteps].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(images, device = self.device)

        encodedImages = mean + std * epsilon  #Reparametrization trick
        return encodedImages, epsilon
    
    @torch.inference_mode()
    def generate(self, numImages,number_inf_steps=None,eta=0,seq_selector="Linear",unet=None):
        '''
        :param number_inf_steps: The number of steps used during inference, by default it's (timesteps)/10
        :param eta: the fraction of variance used, (eta=1) for DDPM and (eta=0) for DDIM
        :param seq_selector: the selection procedure of inference steps, can be Linear or Quadratic
        :param unet: Pass another trained Unet to be used during inference, if not given takes the class's Unet
        '''
        x_Ts = []
        x_T = torch.randn(numImages, 1, 28, 28, device = self.device) #Starting with random noise
        x_Ts.append(self.tensor2numpy(x_T.cpu()))

        if number_inf_steps is None:
            number_inf_steps=int(self.timesteps/10)
        
        if unet is None:
            unet=self.UNet
            
        eta=torch.tensor(eta).to(self.device)

        #Finding sequence of inference timesteps
        if seq_selector=="Linear":
            C = self.timesteps // number_inf_steps
            seq = range(0, self.timesteps, C)

        elif seq_selector=="Quadratic":

            seq = (
                    np.linspace(
                        0, np.sqrt(self.timesteps * 0.8), number_inf_steps
                    )
                    ** 2
                )
            seq = [int(s) for s in list(seq)]

        else:
            print("Invalid seq_seleector, choose between Linear and Quadratic")

        seq_next = [-1] + list(seq[:-1])

        #Inference Loop
        for j1,i1 in tqdm(zip(reversed(seq), reversed(seq_next))):
            
            i=torch.tensor(i1+1,dtype=int).to(self.device)
            j=torch.tensor(j1+1,dtype=int).to(self.device)

            z = torch.randn(numImages, 1, 28, 28, device = self.device) 
            epsilon_theta = unet(x_T, j).sample #Predicted Noise 

            sig=eta*((1-self.alpha_cumprod[i])/(1-self.alpha_cumprod[j])*(1-(self.alpha_cumprod[j])/(self.alpha_cumprod[i]))).sqrt()##DDIM Inference Step
            mean1=((x_T-(1-self.alpha_cumprod[j]).sqrt()*epsilon_theta)/(self.alpha_cumprod[j].sqrt()))
            mean=self.alpha_cumprod[i].sqrt()*mean1+(1-self.alpha_cumprod[i]-sig**2).sqrt()*epsilon_theta 
                
            x_T = mean + z*sig

            x_Ts.append(self.tensor2numpy(x_T.cpu()))

        return x_Ts
    
    @staticmethod
    def tensor2numpy( images):
        images = renorm(images)
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images


        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, help = "Number of timesteps",default=1000)
    parser.add_argument("--beta-start", type = float,default=1e-4)
    parser.add_argument("--beta-end", type = float,default=1e-1)
    parser.add_argument("--log-step", type = int, default = 100)
    parser.add_argument("--checkpoint-step", type = int, default = 195)
    parser.add_argument("--checkpoint", type = str)
    parser.add_argument("--batch-size", type = int,default=64)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--num-epochs", type = int, default = 5)
    parser.add_argument("--seq-selector", type = str, default = "Quadratic")

    args = parser.parse_args()

    dataset = torchvision.datasets.CIFAR10(root = '/home/shorya1835/datasets', download = True,  transform = preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,shuffle = True, drop_last = True, batch_size = args.batch_size, num_workers = 8)

    config = dict(sample_size = 28,
    in_channels = 1,
    out_channels = 1,
    layers_per_block = 3,
    block_out_channels = (32, 64, 128),
    norm_num_groups = 4,
    down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types = (
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ))

    ddim = DDIM(args.beta_start, args.beta_end, args.timesteps,
                config)
    
    ddim.train(dataloader, args.num_epochs, args.checkpoint, 
               args.log_step, args.checkpoint_step, args.lr)

    model = UNet2DModel(**config).to("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load('model.pth')["model"]
    model.load_state_dict(state_dict)
    model.eval()

    img=ddim.generate(numImages=30,eta=0.5,seq_selector=args.seq_selector,unet=model)[-1]

    to_pil_image = transforms.ToPILImage()
    for i,x in enumerate(img):
        image = to_pil_image(x)
        image.save(f'/home/shorya1835/difproj/images_im/image{i+1}.png')
    



