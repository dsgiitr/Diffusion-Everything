import os
import torch
import warnings
import argparse
import torch.utils
import torchvision
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers import UNet2DModel
import torchvision.transforms as T
from Classifier import UNet_Encoder

warnings.filterwarnings("ignore")

def accuracy(inp1, inp2):
    return (inp1 == inp2).sum().item() / len(inp1)

class DDPM():
    def __init__(self, betaStart, betaEnd, timesteps, UNetConfig ,
                 clf_checkpoint = None, unet_checkpoint = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #Initializing Models
        self.UNet = UNet2DModel(**UNetConfig).to(self.device)
        self.clf = UNet_Encoder().to(self.device)

        self.betaStart = betaStart
        self.betaEnd = betaEnd 
        self.timesteps = timesteps

        self.clf_checkpoint = clf_checkpoint
        self.unet_checkpoint = unet_checkpoint
        self.size = UNetConfig["sample_size"]
        self.channels = UNetConfig["in_channels"]

        if unet_checkpoint != None and os.path.isfile(unet_checkpoint):
            self.UNet.load_state_dict(torch.load(unet_checkpoint, map_location = self.device)["model"])
            print("UNet checkpoint loaded !")
        if clf_checkpoint != None and os.path.isfile(clf_checkpoint):
            self.clf.load_state_dict(torch.load(clf_checkpoint, map_location = self.device)["clf"])
            print("Clasifier checkpoint loaded !")

        #DDPM Hyperparameters
        self.betas = torch.linspace(betaStart, betaEnd, timesteps, device = self.device)
        self.alphas = 1 - self. betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.sigmas = self.betas.sqrt()
        self. preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * UNetConfig["in_channels"], [0.5] * UNetConfig["in_channels"])
        ])

        self.renorm = T.Compose([
            T.Normalize([-1] * UNetConfig["out_channels"], [2] * UNetConfig["out_channels"]),
        ])
    def trainCLF(self, numEpochs, dataloader, 
                logStep, checkpointStep, lr):
        self.clf.train()
        self.clf.requires_grad_(True)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        for epoch in range(numEpochs):
            print(f"Epoch [{epoch+1}/{numEpochs}]")
            acc =  0
            for i, (batch, y) in tqdm(enumerate(dataloader), total = len(dataloader)):

                batch = batch.to(self.device)
                ts = torch.randint(0, self.timesteps, (batch.shape[0], ), device = self.device)
                encodedImages, _ = self.addNoise(batch, ts) 
                y = y.to(self.device)
                batch = self.renorm(encodedImages)
                logits = self.clf(batch,ts)
                out = logits.argmax(-1)
                acc += accuracy(out, y)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % logStep == 0 :
                    tqdm.write(f"Step : {i+1} | Loss : {round(loss.item(), 4)}")
                if (i+1) % checkpointStep == 0 :
                    torch.save({
                    "clf" : self.clf.state_dict(), 
                    "timesteps" : self.timesteps
                }, self.clf_checkpoint)
            tqdm.write(f"Accuracy : {round(acc / len(dataloader), 3)}")


    def train(self, dataloader, numEpochs,
             logStep, checkpointStep, lr):
        self.UNet.train()
        self.UNet.requires_grad_(True)
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
                    tqdm.write(f"Step : {i+1} | Loss : {round(loss.item(), 3)}")
                if (i+1) % checkpointStep == 0 :
                    torch.save({
                        "model" : self.UNet.state_dict(), 
                        "beta_start" : self.betaStart,
                        "beta_end" : self.betaEnd,
                        "timesteps" : self.timesteps,
                    }, self.unet_checkpoint)
    
    def addNoise(self, images, timesteps): #Forward process
        mean = self.alpha_cumprod.sqrt()[timesteps].view(-1, 1, 1, 1) * images 
        std = (1 - self.alpha_cumprod).sqrt()[timesteps].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(images, device = self.device)

        encodedImages = mean + std * epsilon  #Reparametrization trick
        return encodedImages, epsilon
    
    def generate(self, numImages, guidanceScale): #Reverse process
        self.UNet.eval()
        self.clf.eval()
        self.clf.requires_grad_(False)
        self.UNet.requires_grad_(False)
        x_Ts = []
        x_T = torch.randn(numImages, self.channels, self.size, self.size, device = self.device) #Starting with random noise
        x_Ts.append(self.tensor2numpy(x_T.cpu()))
        for t in tqdm(torch.arange(self.timesteps - 1, -1, -1, device = self.device)):
            z = torch.randn(numImages, self.channels, self.size, self.size, device = self.device) 
            epsilon_theta = self.UNet(x_T, t).sample #Predicted Noise
            t = t.unsqueeze(0)
            x_T.requires_grad_(True)
            x_T_opt = torch.optim.Adam([x_T], lr=args.lr_clf)  # Optimizer for updating the input
            # Iteratively update the input based on classifier predictions
            acc =  0
            num_steps = 50
            for i in range(num_steps):
                logits = self.clf(self.renorm(x_T), t)
                out = logits.argmax(-1)
                acc += accuracy(out,torch.LongTensor(args.labels).to(self.device))
                loss = F.cross_entropy(logits, torch.LongTensor(args.labels).to(self.device))
                # Backpropagate and update input
                x_T_opt.zero_grad()
                loss.backward()
                x_T_opt.step()
            grads = x_T.grad.data
            x_T.requires_grad_(False)
            t = t.squeeze()
            mean = (1 / self.alphas[t].sqrt()) * (x_T  - ((1 - self.alphas[t])/(1 - self.alpha_cumprod[t]).sqrt()) * epsilon_theta) ##DDPM Inference Step
            x_T = mean + guidanceScale * self.sigmas[t] * grads  +  z * self.sigmas[t] ## added classifier guidance term 
            x_Ts.append(self.tensor2numpy(x_T.cpu()))
        return x_Ts
    
    def tensor2numpy(self, images):
        images = self.renorm(images)
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images

def str_to_int_list(s):
    try:
        return list(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("List must be a comma-separated list of integers.")
        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 500, help = "Number of timesteps")
    parser.add_argument("--beta-start", type = float, default = 1e-4)
    parser.add_argument("--beta-end", type = float, default = 1e-1)
    parser.add_argument("--log-step", type = int, default = 50)
    parser.add_argument("--checkpoint-step", type = int, default = 50)
    parser.add_argument("--unet-checkpoint", default = "unet_mnist_500.ckpt", type = str , help = "Checkpoint path for UNet")
    parser.add_argument("--clf-checkpoint", default = "clf_mnist_500.ckpt", type = str, help="Checkpoint path for Classifier")
    parser.add_argument("--batch-size", type = int, default = 64, help = "Training batch size")
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--num-epochs", type = int, default = 5, help = "Numner of training epochs over complete dataset")
    parser.add_argument("--num-images", type = int, default = 2, help = "Number of images to be generated")
    parser.add_argument("--labels", type=str_to_int_list, required=True, help="Comma-separated list of integers")
    parser.add_argument("--num-epochs-clf", type = int, default = 5)
    parser.add_argument("--lr-clf", type = float, default = 1e-3)
    parser.add_argument("--guidance-scale", type = float, default = 1)
    parser.add_argument("--generate", action = "store_true", help = "Add this to only generate images using model checkpoints")
    parser.add_argument("--output-dir", type = str, default = "images")

    args = parser.parse_args()

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

    ddpm = DDPM(
        betaStart = args.beta_start, 
        betaEnd = args.beta_end, 
        timesteps = args.timesteps, 
        UNetConfig = config, 
        clf_checkpoint = args.clf_checkpoint, 
        unet_checkpoint = args.unet_checkpoint
    )

    if not args.generate :
        dataset = torchvision.datasets.MNIST(root = '../datasets', download = True, transform = ddpm.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, drop_last = True, batch_size = args.batch_size, num_workers = 3)

        ddpm.trainCLF(
            numEpochs = args.num_epochs_clf, 
            dataloader = dataloader,
            logStep = args.log_step, 
            checkpointStep = args.checkpoint_step, 
            lr = args.lr_clf
        )
        
        ddpm.train(
            dataloader = dataloader, 
            numEpochs = args.num_epochs, 
            logStep = args.log_step, 
            checkpointStep = args.checkpoint_step, 
            lr = args.lr
        )

    images = ddpm.generate(args.num_images, args.guidance_scale)[-1]
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(len(images)):
        if images[i].shape[-1] == 1 :
            Image.fromarray(images[i][:, :, 0]).save(os.path.join(args.output_dir, f"image{i+1}.png"))
        else :
            Image.fromarray(images[i]).save(os.path.join(args.output_dir, f"image{i+1}.png"))