import os
import sys
import time
import json 
import torch
from vae import *
import torchvision
import numpy as np
import pandas as pd
from utils import * 
import altair as alt
from PIL import Image
import streamlit as st
import noise_scheduler
from functools import partial
from diffusion import Diffusion
import torchvision.transforms as T
from skimage.restoration import estimate_sigma 

dataset_path = os.environ["DATASET"]
device = os.environ["DEVICE"] if "DEVICE" in os.environ else ("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("./reverse_samplers/")
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed", 
    page_title = "Diffusion Everything", 
    page_icon = "content/favicon.png", 
)
st.page_link("pages/latent_diffusion.py")
st.page_link("pages/visual_diffusion_demos.py")




st.markdown(f"""
 <style>
    {open("style.css").read()}
 </style>
""", unsafe_allow_html = True)

st.title("Diffusion Everything")

st.markdown(open("content/intro.md").read())


link1, link2, link3 = st.columns([0.33, 0.33, 0.33], gap = 'large')


link1.markdown(f'''
    <a class="border" href="#forward-diffusion">
        <h5>Reverse Samplers</h5> 
        <p>
            {open("content/reverse_samplers_intro.md").read()}
        </p>
    </a>''', unsafe_allow_html=True)
link2.markdown(f'''
    <a class="border" href="/latent_diffusion">
        <h5>VAE Latent Diffusion</h5> 
        <p>
            {open("content/latent_diffusion_intro.md").read()}
        </p>
    </a>''', unsafe_allow_html=True)
link3.markdown(f'''
    <a class="border" href="/visual_diffusion_demos">
        <h5>2D and 3D Diffusion</h5> 
        <p>
            {open("content/diffusion_2d.md").read()}
        </p>
    </a>''', unsafe_allow_html=True)

st.markdown(" ")

names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


@st.cache_resource
def get_model(sampler, scheduler):
    models = json.loads(open("models.json").read())
    if scheduler.lower() == "linear" : 
        model_config = models["linear"][sampler]
        unet_config = model_config["unet"]
        timesteps = model_config["timesteps"]
        beta_start = model_config["beta_start"]
        beta_end = model_config["beta_end"]
        checkpoint = model_config["path"]
        model = Diffusion(
            timesteps = timesteps, 
            betaStart = beta_start, 
            betaEnd = beta_end, 
            UNetConfig = unet_config,
            scheduler = scheduler.lower(), 
            checkpoint = checkpoint, 
            device = device, 
        )

    # Add cosine
    return model 


dataset = torchvision.datasets.CIFAR10(root = dataset_path, download = True)
labels = torch.LongTensor(dataset.targets)
idx = torch.arange(len(labels))

def get_random(cls):
    idxs = idx[labels == cls]
    idxs = idxs[torch.randint(0, len(idxs), (1, ))][0].item()
    img = dataset.data[idxs]
    return img

st.subheader("Forward Diffusion")
col1, col2 = st.columns([0.5, 0.5])
col1.markdown(open("content/forward_diffusion.md").read())
col3, col4 = col2.columns([0.5, 0.5])
forward_selected_name = col3.selectbox("Choose a CIFAR-10 category", names, key = "1")
forward_scheduler = col4.selectbox("Select Variance Scheduler", ["Linear", "Cosine"], key = "2")
if forward_scheduler == "Linear" :
    st.session_state["bs"] = col3.number_input("Beta Start", min_value = 0.0, step = 0.001, value = 0.001, format = "%.4f")
    st.session_state["be"] = col4.number_input("Beta End", min_value = 0.0, step = 0.001, value = 0.01, format = "%.4f")

    
st.session_state["fw_prog"] = st.progress(0)    
random_image = st.container()
forward_cols = random_image.columns(11)
if col3.button("Sample Random Image") : 
    class_label = names.index(forward_selected_name)
    start = get_random(class_label)
    forward_cols[0].image(start, caption = "Step 0", use_column_width = True)
    st.session_state["start"] = preprocess(start)
    
snrchart = col3.empty()
show_button = col4.empty()
alpha_chart = col4.empty()

if "chartset" in st.session_state and st.session_state["chartset"] == True:
    snrchart.altair_chart(st.session_state["chart1"], use_container_width = True)
    alpha_chart.altair_chart(st.session_state["chart2"], use_container_width = True)

if show_button.button("Show Forward Process") :
    
    if forward_scheduler == "Cosine" :
        ns = noise_scheduler.Cosine(timesteps = 1000, device = "cpu")
    if forward_scheduler == "Linear" :
        ns = noise_scheduler.Linear(
                timesteps = 1000, 
                beta_start = st.session_state["bs"], 
                beta_end = st.session_state["be"], 
                device = "cpu"
            )

    st.session_state["forward_images"] = []

    for i in range(10):
        st.session_state["fw_prog"].progress(i / 10)
        
        img = ns.forward_process(st.session_state["start"], i * 100)[0].unsqueeze(0)
        img = tensor2numpy(img)[0]
        forward_cols[i].image(img, caption = f"Step {i*100}", use_column_width = True)

        snr = (ns.alpha_cumprod / (1 - ns.alpha_cumprod)).log()

        data = pd.DataFrame({'Timesteps': list(range((i + 1) * 100)), 'log(SNR)': snr[:(i+1) * 100]})
        data1 = pd.DataFrame({'Timesteps': list(range((i + 1) * 100)), 'alpha_bar': ns.alpha_cumprod[:(i+1) * 100]})

        st.session_state["forward_images"].append(img)
        st.session_state["chart1"] = alt.Chart(data).mark_line().encode(
                x='Timesteps',
                y='log(SNR)' ).properties(
                    title = f"log(SNR)",
                )
        st.session_state["chart2"] = alt.Chart(data1).mark_line().encode(
            x = "Timesteps", 
            y = "alpha_bar", 
        ).properties(
                    title = f"alpha cumprod",
                )

        snrchart.altair_chart(st.session_state["chart1"] ,use_container_width = True)
        alpha_chart.altair_chart(st.session_state["chart2"] , use_container_width = True)
        st.session_state["chartset"] = True

        time.sleep(0.1)

    st.session_state["fw_prog"].progress(0.999)

    img = ns.forward_process(st.session_state["start"], 999)[0].unsqueeze(0)
    img = tensor2numpy(img)[0]
    forward_cols[-1].image(img, caption = f"Step 999", use_column_width = True)

    st.session_state["forward_images"].append(img)


st.subheader("Reverse Diffusion")

col5, col6 = st.columns([0.5, 0.5], gap = "large")

samplers = os.listdir("./content/samplers")

sampler_choice = col5.selectbox("Choose Reverse Sampler", samplers)

for s in samplers :
    if s == sampler_choice :
        col5.markdown(open(f"content/samplers/{s}/info.md").read())
        col6.markdown(open(f"content/samplers/{s}/code.md").read())
        break 
    
col7, col8 = col6.columns([0.5,0.5])

selected_name = col7.selectbox("Choose a CIFAR-10 category", names, key = "3")

scheduler = col8.selectbox("Select Variance Scheduler", ["Linear", "Cosine"], key = "4")


st.session_state["steps"] = col7.number_input("Generation Steps", min_value = 0, max_value = 1000, value = 500, step = 10)
st.session_state["lerp"] = col8.number_input("Lerp", min_value = 0.0, step = 0.1, format = "%.1f")

if col7.button("Generate"):
    st.session_state["diffusion"] = get_model(sampler_choice, scheduler)
    progress_bar = st.progress(0)
    images_container = st.container()
    col10, col11 = st.columns([0.75, 0.25], gap = "large")
    with col10 :
        chart_placeholder = st.empty()
    with col11 :
        gif_placeholder = st.empty()
    col13, col14, col15 = st.columns([0.3, 0.3, 0.4])
    with col13 :
        another_gif = st.empty()

    num_images = 11

    cols = images_container.columns(num_images)
    arr = [] 
    count = 0

    def callback(epsilon_theta, mean, old_x_T,  x_T, T, step_idx):
        global count
        step = (st.session_state["steps"] - T - 1).item()
        if step == step_idx[count] :
            cols[count].image(x_T[0], caption=f"Step {step}", use_column_width = True) 
            count += 1
        arr.append(estimate_sigma(x_T[0]))
        progress_bar.progress(step / (st.session_state["steps"] - 1))

        data = pd.DataFrame({
                "Noise" : arr, 
                "X" : list(range(len(arr)))
            })

        chart_placeholder.altair_chart(
                alt.Chart(data).mark_line().encode(
                    y=alt.Y('Noise', title='Standard Deviation'),
                    x=alt.X('X', title='Timesteps')
                ).properties(
                    title = f"Estimated Noise in the image",
                ), 
                use_container_width = True, 
            )

        if step % 10 == 0 :
            gif_placeholder.image(x_T[0], use_column_width = True, caption = f"Live Generations (Step {step})")
        if step == st.session_state["steps"] - 1 :
            gif_placeholder.image(x_T[0], use_column_width = True, caption = f"Live Generations (Step {step})")

    step_idx = torch.linspace(0, st.session_state["steps"] - 1, 11).long().tolist()
    partial_callback = partial(callback, step_idx = step_idx)
    if st.session_state["steps"] <= 1000 :
        st.session_state["diffusion"].sampler.reverse(
                    numImages = 1, 
                    labels = torch.tensor([names.index(selected_name)], 
                    device = st.session_state["diffusion"].device), 
                    steps = st.session_state["steps"], 
                    lerp = st.session_state["lerp"], 
                    streamlit_callback = partial_callback
                )
    st.success("Generation complete!")

