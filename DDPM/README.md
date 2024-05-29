## Denoising Diffusion Probabilistic Model - DDPM 

Simple implementation of DDPM in one single python script with training and inference

### Usage 
---

```
usage: ddpm.py [-h] [--timesteps TIMESTEPS] [--beta-start BETA_START] [--beta-end BETA_END] [--log-step LOG_STEP]
               [--checkpoint-step CHECKPOINT_STEP] [--checkpoint CHECKPOINT] [--batch-size BATCH_SIZE] [--lr LR] [--num-epochs NUM_EPOCHS]
               [--num-images NUM_IMAGES] [--generate] [--config CONFIG] [--output-dir OUTPUT_DIR]

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
```  

* First download the model checkpoint 
```bash
wget https://huggingface.co/P3g4su5/DDPM-UNet/resolve/main/ddpm.ckpt?download=true -O ddpm.ckpt
``` 
* Then run the following script 
```bash
#!/bin/bash

python3 ddpm.py \
	--beta-start 1e-4 \
	--beta-end 2e-2 \
	--timesteps 1000 \
	--num-images 50 \
	--checkpoint "ddpm.ckpt" \
	--output-dir "sample-images" \
	--config "config.json" \
	--generate 
```
* Since this is vanilla DDPM, 1000 steps will take quite a lot of time to generate so I suggest running inference on a GPU


