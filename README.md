# Diffusion Everything  
Diffusion models as of now are the go-to models in the field of Generative AI be it images or video. This project aims to help understand the process with three interactive demos. 
1. Diffusion on CIFAR-10 : We trained multiple variants of diffusion models along with option to choose from different reverse samplers from DDPM sampler to latest DPM2++ sampler. You will find code snippets as well as simple description of all the reverse samples. 
2. Latent Diffusion with MNIST : In this part we first train a simple VAE on MNIST dataset to maps the dataset to a 2D distribution. Then we conditionally sample points from the 2D dataset using a diffusion model. 
3. Diffusion on 2D and 3D datasets : To gain an intuitive understanding of sampling process we can train diffusion models on 2D shapes and custom drawings, then sample thousands of points using diffusion model to recreate the complete dataset. 

## Setup and Run
```bash 
git clone https://github.com/dsgiitr/Diffusion-Everything/
cd Diffusion-Everything 
mkdir ~/.streamlit 
mv config.toml ~/.streamlit/config.toml
pip3 install -r requirements.txt 
```
Before running make sure you download the pre-trained models using and set these two environment variables accordingly. We highly suggest using a device with a decent GPU for seamless experience. 
```
export DATASET=/home/user/datasets
export DEVICE=cuda
chmod +x ./download.sh
./download.sh
```
Now, to run the demos 
```
streamlit run diffusion-everything.py --server.port 80 --client.showSidebarNavigation False
```
## Usage 
Apart from the demos you can use  the codebase from VAE and Diffusion to run and train your own custom models. 
### Train Diffusion Model 
```
python3 diffusion.py --help
usage: diffusion.py [-h] [--timesteps TIMESTEPS] [--beta-start BETA_START] [--beta-end BETA_END] [--log-step LOG_STEP]
                    [--checkpoint-step CHECKPOINT_STEP] [--checkpoint CHECKPOINT] [--batch-size BATCH_SIZE] [--lr LR] [--num-epochs NUM_EPOCHS]
                    [--num-images NUM_IMAGES] [--generate] [--config CONFIG] [--output-dir OUTPUT_DIR] [--scheduler SCHEDULER]

options:
  -h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        Number of timesteps
  --beta-start BETA_START
  --beta-end BETA_END
  --log-step LOG_STEP
  --checkpoint-step CHECKPOINT_STEP
  --checkpoint CHECKPOINT
                        Checkpoint path for UNet
  --batch-size BATCH_SIZE
                        Training batch size
  --lr LR               Learning rate
  --num-epochs NUM_EPOCHS
                        Numner of training epochs over complete dataset
  --num-images NUM_IMAGES
                        Number of images to be generated (if any)
  --generate            Add this to only generate images using model checkpoints
  --config CONFIG       Path of UNet config file in json format
  --output-dir OUTPUT_DIR
  --scheduler SCHEDULER
```
Apart from command line interface the model can also me used by importing as a module 
```python
from diffusion import Diffusion 

model = Diffusion(
	betaStart = 1e-3, 
	betaEnd = 1e-2, 
	timesteps = 1000, 
	UNetConfig = "config.json", 
	scheduler = "linear", 
	checkpoint = "linear.ckpt", 
	device = "cuda"
)
model.train(
	dataloader = dataloader, 
	numEpochs = 10, 
	logStep = 50, 
	checkpointStep = 50, 
	lr = 1e-4
)

#Genrate Images
images = model.generate(
	num_images = 5, 
	labels = [0,1, 2, 3, 4]
)
```

## Contribution Guide 
Furthur contribution can be made by adding new reverse samplers and pre-trained models in the following format. 



