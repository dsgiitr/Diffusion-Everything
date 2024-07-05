import os
import json
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
from utils import noise_scheduler
from Classifier import UNet_Encoder

warnings.filterwarnings("ignore")

def accuracy(inp1, inp2):
    return (inp1 == inp2).sum().item() / len(inp1)

class DDPM():
    def __init__(self,
    betaStart,
    betaEnd,
    timesteps,
    UNetConfig,
    scheduler = "cosine",
    clf_checkpoint = None,
    unet_checkpoint = None,
    device = None
    ):
        self.clf_checkpoint = clf_checkpoint
        self.unet_checkpoint = unet_checkpoint
        self.timesteps = timesteps
        if device == None :
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else :
            self.device = device
        self.UNet = UNet2DModel(**UNetConfig).to(self.device)
        self.clf = UNet_Encoder().to(self.device)
        self.load_checkpoint_unet(unet_checkpoint)
        self.load_checkpoint_classifier(clf_checkpoint)
        self.img_shape = [UNetConfig["in_channels"], UNetConfig["sample_size"], UNetConfig["sample_size"]]

        #DDPM Schedulers
        if scheduler == "linear" : 
            self.noise_scheduler = noise_scheduler.Linear(betaStart, betaEnd, timesteps, self.device)
        elif scheduler == "cosine" :    
            self.noise_scheduler = noise_scheduler.Cosine(timesteps, self.device)

        self. preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * UNetConfig["in_channels"], [0.5] * UNetConfig["in_channels"])
        ])

        self.renorm = T.Compose([
            T.Normalize([-1] * UNetConfig["out_channels"], [2] * UNetConfig["out_channels"]),
        ])

    def load_checkpoint_unet(self, unet_checkpoint):
        if unet_checkpoint != None and os.path.isfile(unet_checkpoint):
            self.UNet.load_state_dict(torch.load(unet_checkpoint, map_location = self.device)["model"])
            print("UNet checkpoint loaded !")
    
    def load_checkpoint_classifier(self, clf_checkpoint):
        if clf_checkpoint != None and os.path.isfile(clf_checkpoint):
            self.clf.load_state_dict(torch.load(clf_checkpoint, map_location = self.device)["clf"])
            print("Clasifier checkpoint loaded !")
    
    def save_checkpoint(self):
        torch.save({
            "model" : self.UNet.state_dict(),
        }, self.unet_checkpoint)

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
                encodedImages, _= self.noise_scheduler.forward_process(batch, ts)
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

    def train(self, 
        dataloader, 
        numEpochs,
        logStep, 
        checkpointStep, 
        lr, 
        ema = True, ):
        
        self.UNet.train()
        if ema :
            swa_model = torch.optim.swa_utils.AveragedModel(self.UNet, 
                                                            device = self.device, 
                                                            multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        optimizer = torch.optim.AdamW(self.UNet.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = numEpochs * len(dataloader), eta_min = 1e-5)

        for epoch in range(numEpochs):
            print(f"Epoch [{epoch+1}/{numEpochs}]")
            for i, (img, y) in tqdm(enumerate(dataloader), total = len(dataloader)):
                img = img.to(self.device)
                y = y.to(self.device)
                # print("Here")
                ts = torch.randint(0, self.timesteps, (img.shape[0], ), device = self.device)
                encodedImages, epsilon = self.noise_scheduler.forward_process(img, ts)
                predictedNoise = self.UNet(encodedImages, ts, y).sample
                # print("Done forward")
                loss = F.mse_loss(predictedNoise, epsilon)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if ema :
                    swa_model.update_parameters(self.UNet)
                scheduler.step()

                if (i+1) % logStep == 0 :
                    tqdm.write(f"Step : {i+1} | Loss : {loss.item()}")
                if (i+1) & checkpointStep == 0 :
                    self.save_checkpoint()

    def generate(self, numImages, guidanceScale, labels,streamlit_callback = None): #Reverse process
        self.UNet.eval()
        self.clf.eval()
        self.clf.requires_grad_(False)
        self.UNet.requires_grad_(False)
        ns = self.noise_scheduler
        x_Ts = []
        x_T = torch.randn(numImages, *self.img_shape, device = self.device) #Starting with random noise
        x_Ts.append(self.tensor2numpy(x_T.cpu()))
        y = torch.LongTensor(labels).to(self.device)
        
        for t in tqdm(torch.arange(self.timesteps - 1, 0, -1, device = self.device)):
            z = torch.randn(numImages, *self.img_shape, device = self.device) 
            epsilon_theta = self.UNet(x_T, t, y).sample # Predicted Noise
            t = t.unsqueeze(0)
            x_T.requires_grad_(True)
            x_T_opt = torch.optim.Adam([x_T], lr=args.lr_clf)  # Optimizer for updating the input
            # Iteratively update the input based on classifier predictions
            acc =  0
            num_steps = 50
            for i in range(num_steps):
                logits = self.clf(self.renorm(x_T), t)
                out = logits.argmax(-1)
                acc += accuracy(out, y)
                loss = F.cross_entropy(logits, y)
                # Backpropagate and update input
                x_T_opt.zero_grad()
                loss.backward()
                x_T_opt.step()
            tqdm.write(f"Step : {i+1} | Loss : {round(loss.item(), 4)}")
            tqdm.write(f"Accuracy : {round(acc/num_steps, 3)}")
            print("predicted labels :",out)
            grads = x_T.grad.data
            x_T.requires_grad_(False)
            t = t.squeeze()
            mean = (1 / ns.alpha[t].sqrt()) * (x_T - ((1 - ns.alpha[t])/(1 - ns.alpha_cumprod[t]).sqrt()) * epsilon_theta)
            old_x_T = x_T 
            x_T = mean + guidanceScale * ns.sigma[t] * grads  +  z * ns.sigma[t] 
            if streamlit_callback :
                streamlit_callback(epsilon_theta, mean, old_x_T, self.tensor2numpy(x_T.cpu()), t)
            x_Ts.append(self.tensor2numpy(x_T.cpu()))
        return x_Ts

    def fast_generate(self, numImages, guidanceScale, labels, steps, lerp = 1,streamlit_callback = None): 
        self.UNet.eval()
        self.clf.eval()
        self.clf.requires_grad_(False)
        self.UNet.requires_grad_(False)
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
        x_Ts.append(self.tensor2numpy(x_T.cpu()))
        y = torch.LongTensor(labels).to(self.device)
        
        for t in tqdm(torch.arange(self.timesteps - 1, 0, -1, device = self.device)):
            z = torch.randn(numImages, *self.img_shape, device = self.device) 
            epsilon_theta = self.UNet(x_T, idx[t], y).sample 
            t = t.unsqueeze(0)
            x_T.requires_grad_(True)
            x_T_opt = torch.optim.Adam([x_T], lr=args.lr_clf)  
            acc =  0
            num_steps = 50
            for i in range(num_steps):
                logits = self.clf(self.renorm(x_T), t)
                out = logits.argmax(-1)
                acc += accuracy(out, y)
                loss = F.cross_entropy(logits, y)
                x_T_opt.zero_grad()
                loss.backward()
                x_T_opt.step()
            tqdm.write(f"Step : {i+1} | Loss : {round(loss.item(), 4)}")
            tqdm.write(f"Accuracy : {round(acc/num_steps, 3)}")
            print("predicted labels :",out)
            grads = x_T.grad.data
            x_T.requires_grad_(False)
            t = t.squeeze()
            mean = (1 / alpha[t].sqrt()) * (x_T - ((1 - alpha[t])/(1 - alpha_cumprod[t]).sqrt()) * epsilon_theta)
            old_x_T = x_T 
            x_T = mean + guidanceScale * sigma[t] * grads  +  z * sigma[t] 
            if streamlit_callback :
                streamlit_callback(epsilon_theta, mean, old_x_T, self.tensor2numpy(x_T.cpu()), t)
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
    parser.add_argument("--timesteps", type = int, default = 100, help = "Number of timesteps")
    parser.add_argument("--beta-start", type = float, default = 1e-4)
    parser.add_argument("--beta-end", type = float, default = 1e-1)
    parser.add_argument("--log-step", type = int, default = 50)
    parser.add_argument("--checkpoint-step", type = int, default = 50)
    parser.add_argument("--unet-checkpoint", default = "unet_mnist_500.ckpt", type = str , help = "Checkpoint path for UNet")
    parser.add_argument("--clf-checkpoint", default = "clf_mnist_500.ckpt", type = str, help="Checkpoint path for Classifier")
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--num-epochs", type = int, default = 5)
    parser.add_argument("--num-images", type = int, default = 10)
    parser.add_argument("--labels", type=str_to_int_list, required=True, help="Comma-separated list of integers")
    parser.add_argument("--num-epochs-clf", type = int, default = 5)
    parser.add_argument("--lr-clf", type = float, default = 1e-3)
    parser.add_argument("--guidance-scale", type = float, default = 1)
    parser.add_argument("--output-dir", type = str, default = "images")
    parser.add_argument("--scheduler", type = str,  default = "cosine")
    parser.add_argument("--config", type = str, help = "Path of UNet config file in json format")
    parser.add_argument("--generate", action = "store_true", help = "Add this to only generate images using model checkpoints")

    args = parser.parse_args()

    with open(args.config) as file :
        config = json.load(file)

    ddpm = DDPM(
        betaStart = args.beta_start, 
        betaEnd = args.beta_end, 
        timesteps = args.timesteps, 
        UNetConfig = config,
        scheduler = args.scheduler, 
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

    images = ddpm.generate(args.num_images, args.guidance_scale, args.labels)[-2]
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(len(images)):
        if images[i].shape[-1] == 1 :
            Image.fromarray(images[i][:, :, 0]).save(os.path.join(args.output_dir, f"image{i+1}.png"))
        else :
            Image.fromarray(images[i]).save(os.path.join(args.output_dir, f"image{i+1}.png"))