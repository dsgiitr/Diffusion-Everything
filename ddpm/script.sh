#!/bin/bash

python3 ddpm.py \
	--beta-start 1e-4 \
	--beta-end 2e-2 \
	--timesteps 1000 \
	--num-images 50 \
	--checkpoint "ddpm.ckpt" \
	--output-dir "images" \
	--config "config.json" \
	--generate 
