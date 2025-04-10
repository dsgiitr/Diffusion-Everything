#!/bin/sh
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download P3g4su5/Diffusion-Everything --local-dir ./checkpoints/
