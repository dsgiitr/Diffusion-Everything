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
