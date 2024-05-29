import os
import json
import torch
import warnings
import argparse
import torch.utils
import torchvision
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers import UNet2DModel
import torchvision.transforms as T

warnings.filterwarnings("ignore")


class DDPM():
    def __init__(self, betaStart, betaEnd,
                 timesteps, UNetConfig, checkpoint = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.UNet = UNet2DModel(**UNetConfig).to(self.device)
        self.betaStart = betaStart
        self.betaEnd = betaEnd 
        self.timesteps = timesteps
        self.size = UNetConfig["sample_size"]
        self.channels = UNetConfig["in_channels"]
        self.checkpoint = checkpoint
        if checkpoint != None and os.path.isfile(checkpoint):
            self.UNet.load_state_dict(torch.load(checkpoint, map_location = self.device)["model"])
            print("Checkpoint Loaded")

        #DDPM Hyperparameters
        self.betas = torch.linspace(betaStart, betaEnd, timesteps, device = self.device)
        self.alphas = 1 - self. betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.sigmas = self.betas.sqrt()
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * UNetConfig["in_channels"], [0.5] * UNetConfig["in_channels"])
        ])

        self.renorm = T.Compose([
            T.Normalize([-1] * UNetConfig["out_channels"], [2] * UNetConfig["out_channels"]),
        ])

    def train(self, dataloader, numEpochs,
             logStep, checkpointStep, lr):
        self.UNet.train()
        optimizer = torch.optim.Adam(self.UNet.parameters(), lr = lr)
        for epoch in range(numEpochs):
            print(f"Epoch [{epoch+1}/{numEpochs}]")
            for i, (batch, y) in tqdm(enumerate(dataloader), total = len(dataloader)):
                batch = batch.to(self.device)

                ts = torch.randint(0, self.timesteps, (batch.shape[0], ), device = self.device)
                encodedImages, epsilon = self.addNoise(batch, ts)
                predictedNoise = self.UNet(encodedImages, ts).sample
                optimizer.zero_grad()
                loss = F.mse_loss(predictedNoise, epsilon)
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
                    }, self.checkpoint)
            images = self.generate(args.num_images)[-1]
            for i in range(len(images)):
                if images[i].shape[-1] == 1 :
                    Image.fromarray(images[i][:, :, 0]).save(os.path.join("images", f"image{i+1}.png"))
                else :
                    Image.fromarray(images[i]).save(os.path.join("images", f"image{i+1}.png"))
    
    def addNoise(self, images, timesteps): #Forward process
        mean = self.alpha_cumprod.sqrt()[timesteps].view(-1, 1, 1, 1) * images 
        std = (1 - self.alpha_cumprod).sqrt()[timesteps].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(images, device = self.device)

        encodedImages = mean + std * epsilon  #Reparametrization trick
        return encodedImages, epsilon
    
    @torch.inference_mode()
    def generate(self, numImages): #Reverse process
        x_Ts = []
        x_T = torch.randn(numImages, self.channels, self.size, self.size, device = self.device) #Starting with random noise
        x_Ts.append(self.tensor2numpy(x_T.cpu()))

        for t in tqdm(torch.arange(self.timesteps - 1, -1, -1, device = self.device)):
            z = torch.randn(numImages, self.channels, self.size, self.size, device = self.device) 
            epsilon_theta = self.UNet(x_T, t).sample # Predicted Noise

            mean = (1 / self.alphas[t].sqrt()) * (x_T - ((1 - self.alphas[t])/(1 - self.alpha_cumprod[t]).sqrt()) * epsilon_theta) ##DDPM Inference Step
            
            x_T = mean + z * self.sigmas[t]
            x_Ts.append(self.tensor2numpy(x_T.cpu()))
        return x_Ts
    
    def tensor2numpy(self, images):
        images = self.renorm(images)
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images

        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default =100, help = "Number of timesteps")
    parser.add_argument("--beta-start", type = float, default = 1e-4)
    parser.add_argument("--beta-end", type = float, default = 1e-2)
    parser.add_argument("--log-step", type = int, default = 50)
    parser.add_argument("--checkpoint-step", type = int, default = 50)
    parser.add_argument("--checkpoint", type = str, default = "ddpm.ckpt", help = "Checkpoint path for UNet")
    parser.add_argument("--batch-size", type = int, default = 128, help = "Training batch size")
    parser.add_argument("--lr", type = float, default = 2e-4, help = "Learning rate")
    parser.add_argument("--num-epochs", type = int, default = 5, help = "Numner of training epochs over complete dataset")
    parser.add_argument("--num-images", type = int, default = 10, help = "Number of images to be generated (if any)")
    parser.add_argument("--generate", action = "store_true", help = "Add this to only generate images using model checkpoints")
    parser.add_argument("--config", type = str, help = "Path of UNet config file in json format")
    parser.add_argument("--output-dir", type = str, default = "images")

    args = parser.parse_args()

    with open(args.config) as file :
        config = json.load(file)

    ddpm = DDPM(
        betaStart = args.beta_start, 
        betaEnd = args.beta_end, 
        timesteps = args.timesteps, 
        UNetConfig = config,
        checkpoint = args.checkpoint)


    if not args.generate :
        dataset = torchvision.datasets.CIFAR10(root = '../datasets', download = True, transform = ddpm.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, drop_last = True, batch_size = args.batch_size, num_workers = 3)
        
        ddpm.train(
            dataloader = dataloader, 
            numEpochs = args.num_epochs, 
            logStep = args.log_step, 
            checkpointStep = args.checkpoint_step, 
            lr = args.lr)
    
    images = ddpm.generate(args.num_images)[-1] #Saving final denoised images

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(len(images)):
        if images[i].shape[-1] == 1 :
            Image.fromarray(images[i][:, :, 0]).save(os.path.join(args.output_dir, f"image{i+1}.png"))
        else :
            Image.fromarray(images[i]).save(os.path.join(args.output_dir, f"image{i+1}.png"))