## Classifier-Guidance on DDPM 

Simple implementation of Classifier Guidance on DDPM in one single python script with training and inference

### Usage 
---

```
usage: classifier_guidance_mnist.py [-h] [--timesteps TIMESTEPS] [--beta-start BETA_START] [--beta-end BETA_END] [--log-step LOG_STEP]
               [--checkpoint-step CHECKPOINT_STEP] [--unet-checkpoint UNET_CHECKPOINT] [--clf-checkpoint CLF_CHECKPOINT]
               [--batch-size BATCH_SIZE] [--lr LR] [--num-epochs NUM_EPOCHS] [--num-images NUM_IMAGES] [--labels LABELS]
               [--num-epochs-clf NUM_EPOCHS_CLF] [--lr-clf LR_CLF] [--guidance-scale GUIDANCE_SCALE] [--generate] 
               [--output-dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        Number of timesteps
  --beta-start BETA_START
  --beta-end BETA_END
  --log-step LOG_STEP
  --checkpoint-step CHECKPOINT_STEP
  --unet-checkpoint UNET_CHECKPOINT
                        Checkpoint path for UNet
  --clf-checkpoint CLF_CHECKPOINT
                        Checkpoint path for Classifier
  --batch-size BATCH_SIZE
                        Training batch size
  --lr LR               Learning rate
  --num-epochs NUM_EPOCHS
                        Numner of training epochs over complete dataset
  --num-images NUM_IMAGES
                        Number of images to be generated
  --labels LABELS
                        Labels for images to be generated
  --num-epochs-clf NUM_EPOCHS_CLF
  --lr-clf LR_CLF
  --guidance-scale GUIDANCE_SCALE
  --generate            Add this to only generate images using model checkpoints
  --output-dir OUTPUT_DIR
```  

* First download the model checkpoint 
```bash
wget https://huggingface.co/pranav-5644/classifier_guidance/resolve/main/clf_mnist_500.ckpt?download=true -O clf_mnist_500.ckpt
wget https://huggingface.co/pranav-5644/classifier_guidance/resolve/main/unet_mnist_500.ckpt?download=true -O unet_mnist_500.ckpt
``` 
* Then run the following script 
```bash
#!/bin/bash

!python3 classifier_guidance_mnist.py \
  --beta-start 1e-4 \
  --beta-end 2e-2 \
  --timesteps 500 \
  --num-images 10 \
  --output-dir "sample_images" \
  --unet-checkpoint "unet_mnist_500.ckpt" \
  --clf-checkpoint "clf_mnist_500.ckpt" \
  --labels 5,6,7,0,3,1,2,7,5,4 \
  --guidance-scale 3 \
  --generate
```
* Since this is Classifier-Guidance on vanilla DDPM, 500 steps will take quite a lot of time to generate so I suggest running inference on a GPU


