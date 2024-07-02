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
from diffusers import UNet2DConditionModel
import torchvision.transforms as T
from utils import noise_scheduler

warnings.filterwarnings("ignore")


class CFG():
    def __init__(self, 
        timesteps, 
        UNetConfig, 
        scheduler = "cosine", 
        betaStart = None, 
        betaEnd = None, 
        checkpoint = None,
        device = None, 
    ):
        self.checkpoint = checkpoint
        self.timesteps = timesteps
        if device == None :
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else :
            self.device = device 

        self.UNet = UNet2DConditionModel(**UNetConfig).to(self.device)
        self.load_checkpoint(checkpoint)


        self.img_shape = [UNetConfig["in_channels"], UNetConfig["sample_size"], UNetConfig["sample_size"]]

        #DDPM Schedulers
        if scheduler == "linear" : 
            self.noise_scheduler = noise_scheduler.Linear(betaStart, betaEnd, timesteps, self.device)
        elif scheduler == "cosine" :
            self.noise_scheduler = noise_scheduler.Cosine(timesteps, self.device)

        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * UNetConfig["in_channels"], [0.5] * UNetConfig["in_channels"])
        ])

        self.renorm = T.Compose([
            T.Normalize([-1] * UNetConfig["out_channels"], [2] * UNetConfig["out_channels"]),
        ])


    def load_checkpoint(self ,checkpoint):
        if checkpoint != None and os.path.isfile(checkpoint) :
            self.UNet.load_state_dict(torch.load(checkpoint, map_location = self.device)["model"])
            print("Checkpoint Loaded !")

    def save_checkpoint(self):
        torch.save({
            "model" : self.UNet.state_dict(),
        }, self.checkpoint)

    def train(self, 
        dataloader, 
        numEpochs,
        logStep, 
        checkpointStep, 
        lr, 
        uncondition_rate = 0.2, 
        ema = True, 
    ):
        
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
                y = F.one_hot(y, num_classes = 10).to(self.device).unsqueeze(1).float()
                keep_condition = torch.bernoulli(torch.ones(img.size(0)) - uncondition_rate).to(self.device)  # generates ones with probability 1 - uncondition_rate
                y *= keep_condition.view(-1, 1, 1)

                ts = torch.randint(0, self.timesteps, (img.shape[0], ), device = self.device)
                encodedImages, epsilon = self.noise_scheduler.forward_process(img, ts)
                predictedNoise = self.UNet(encodedImages, ts, encoder_hidden_states=y).sample
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


    
    @torch.inference_mode()
    def generate(self, num_images, gen_class, w, streamlit_callback = None): 

        condition_null = torch.zeros(num_images, 1, 10, device = self.device)
        if gen_class is not None:
            condition = F.one_hot(torch.ones(num_images, dtype=int) * gen_class, num_classes = 10).to(self.device).unsqueeze(1).float()
        ns = self.noise_scheduler
        x_Ts = []
        x_T = torch.randn(num_images, *self.img_shape, device = self.device) 
        x_Ts.append(self.tensor2numpy(x_T))

        for t in tqdm(torch.arange(self.timesteps - 1, -1, -1, device = self.device)):

            epsilon_theta_null = self.UNet(x_T, t, encoder_hidden_states = condition_null).sample 
            if gen_class is not None:
                epsilon_theta_condition = self.UNet(x_T, t, encoder_hidden_states = condition).sample
                epsilon_theta = (w+1) * epsilon_theta_condition - epsilon_theta_null * w
            else :
                epsilon_theta = epsilon_theta_null
            
            old_x_T = x_T
            mean = (1 / ns.alpha[t].sqrt()) * (x_T - ((1 - ns.alpha[t])/(1 - ns.alpha_cumprod[t]).sqrt()) * epsilon_theta) 
            x_T = self.sample(mean, ns.sigma[t])

            if streamlit_callback :
                streamlit_callback(epsilon_theta, mean, self.tensor2numpy(old_x_T), self.tensor2numpy(x_T.cpu()), t)

            x_Ts.append(self.tensor2numpy(x_T.cpu()))
        return x_Ts

    @staticmethod
    def sample(mean, std):
        return mean + std * torch.randn_like(mean)
    
    def tensor2numpy(self, images):
        images = self.renorm(images.detach().cpu())
        images = images.permute(0, 2, 3, 1)
        images = torch.clamp(images, 0, 1).numpy()
        images = (images * 255).astype('uint8')
        return images

        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 100, help = "Number of timesteps")
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
    parser.add_argument("--scheduler", type = str,  default = "cosine")

    args = parser.parse_args()

    with open(args.config) as file :
        config = json.load(file)

    ddpm = CFG(
        betaStart = args.beta_start, 
        betaEnd = args.beta_end, 
        timesteps = args.timesteps, 
        UNetConfig = config,
	    scheduler = args.scheduler, 
        checkpoint = args.checkpoint
    )


    if not args.generate :
        dataset = torchvision.datasets.CIFAR10(root = '/scratch/shorya_s.iitr/datasets', transform = ddpm.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, drop_last = True, batch_size = args.batch_size, num_workers = 4)
        
        ddpm.train(
            dataloader = dataloader, 
            numEpochs = args.num_epochs, 
            logStep = args.log_step, 
            checkpointStep = args.checkpoint_step, 
            lr = args.lr)
    
    images = ddpm.generate(args.num_images, 2, 0.2)[-2] #Saving final denoised images

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(len(images)):
        if images[i].shape[-1] == 1 :
            Image.fromarray(images[i][:, :, 0]).save(os.path.join(args.output_dir, f"image{i+1}.png"))
        else :
            Image.fromarray(images[i]).save(os.path.join(args.output_dir, f"image{i+1}.png"))

