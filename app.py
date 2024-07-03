import streamlit as st
import time
from PIL import Image
import json 
import altair as alt
import numpy as np
import torch
from ddpm import DDPM
import numpy as np
from utils import noise_scheduler
import torchvision
import pandas as pd
from skimage.restoration import estimate_sigma 


st.set_page_config(layout="wide")
menu = ["DDPM - Class Conditioned", "Option 2", "Option 3"]
choice = st.sidebar.selectbox("Menu", menu)

st.title("Diffusion Everything")

@st.cache_resource
def load_models():
    ddpm_cosine = DDPM(
        timesteps = 1000, 
        UNetConfig = json.loads(open("ddpm/config.json").read()),
        scheduler = "cosine", 
        checkpoint = "checkpoints/ddpm-cosine.final", 
        device = "cuda:0", 
    )
    ddpm_linear = DDPM(
        timesteps = 1000, 
        betaStart = 1e-4, 
        betaEnd = 2e-2, 
        UNetConfig = json.loads(open("ddpm/config.json").read()),
        scheduler = "linear", 
        checkpoint = "checkpoints/ddpm-linear.final", 
        device = "cuda:0", 
    )
    return ddpm_cosine, ddpm_linear

names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

ddpm_cosine, ddpm_linear = load_models()

dataset = torchvision.datasets.CIFAR10(root = "./datasets")
labels = torch.LongTensor(dataset.targets)
idx = torch.arange(len(labels))
def get_random(cls):
    idxs = idx[labels == cls]
    idxs = idxs[torch.randint(0, len(idxs), (1, ))][0].item()
    img = dataset.data[idxs]
    return img


if choice == "DDPM - Class Conditioned":
    st.header("DDPM - Class Conditioned")
    st.subheader("Forward Diffusion")
    st.latex(r"x_t = \sqrt{\bar{\alpha_t}} * x_0 + \sqrt{1 - \bar{\alpha_t}} * \epsilon_0")
    forward_selected_name = st.selectbox("Choose a CIFAR-10 category", names, key = "1")
    forward_scheduler = st.radio("Select Variance Scheduler", ["Linear", "Cosine"], key = "2")
    beta_start, beta_end = st.columns(2)
    with beta_start :
        st.session_state["bs"] = st.number_input("Beta Start", min_value = 0.0, step = 0.001, format = "%.4f")
        
    with beta_end :
        st.session_state["be"] = st.number_input("Beta End", min_value = 0.0, step = 0.001, format = "%.4f")
    st.session_state["fw_prog"] = st.progress(0)
    random_image = st.container()
    forward_cols = random_image.columns(11)
    if st.button("Random Image") : 
        class_label = names.index(forward_selected_name)
        start = get_random(class_label)
        forward_cols[0].image(start, caption = "Step 0", width = 80)
        st.session_state["start"] = ddpm_cosine.preprocess(start)
    
    if st.button("Show Forward Process") :
        if forward_scheduler == "Cosine" :
            ns = noise_scheduler.Cosine(timesteps = 1000, device = "cpu")
            for i in range(10):
                st.session_state["fw_prog"].progress(i / 10)
                img = ns.forward_process(st.session_state["start"], i * 100)[0].unsqueeze(0)
                img = ddpm_cosine.tensor2numpy(img)[0]
                forward_cols[i].image(img, caption = f"Step {i*100}", width = 80)
            st.session_state["fw_prog"].progress(0.99)
            img = ns.forward_process(st.session_state["start"], 999)[0].unsqueeze(0)
            img = ddpm_linear.tensor2numpy(img)[0]
            forward_cols[-1].image(img, caption = f"Step 999", width = 80)
        if forward_scheduler == "Linear" :
            ns = noise_scheduler.Linear(timesteps = 1000, beta_start = st.session_state["bs"], beta_end = st.session_state["be"], device = "cpu")
            for i in range(10):
                st.session_state["fw_prog"].progress(i / 10)
                img = ns.forward_process(st.session_state["start"], i * 100)[0].unsqueeze(0)
                img = ddpm_linear.tensor2numpy(img)[0]
                forward_cols[i].image(img, caption = f"Step {i*100}", width = 80)
            st.session_state["fw_prog"].progress(0.99)
            img = ns.forward_process(st.session_state["start"], 999)[0].unsqueeze(0)
            img = ddpm_linear.tensor2numpy(img)[0]
            forward_cols[-1].image(img, caption = f"Step 999", width = 80)
    
    
    st.subheader("Reverse Diffusion")
    st.latex(r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_{\theta}(x_t, t)) - \sigma_t z")
    
    selected_name = st.selectbox("Choose a CIFAR-10 category", names, key = "3")

    scheduler = st.radio("Select Variance Scheduler", ["Linear", "Cosine"], key = "4")

    if scheduler == "Linear" :
        ddpm = ddpm_linear
    elif scheduler == "Cosine" :
        ddpm = ddpm_cosine 
    
    st.session_state["steps"] = st.number_input("Generation Steps", min_value = 0, step = 1)
    st.session_state["lerp"] = st.number_input("Lerp", min_value = 0.0, step = 0.1, format = "%.1f")    
    if st.button("Generate"):
        progress_bar = st.progress(0)
        images_container = st.container()
        col1, col2 = st.columns([0.75, 0.25])
        with col1 :
            chart_placeholder = st.empty()
        with col2 :
            gif_placeholder = st.empty()
        col3, col4, col5 = st.columns([0.3, 0.3, 0.4])
        with col3 :
            another_gif = st.empty()

        num_images = 11

        cols = images_container.columns(num_images)
        arr = [] 
        
        def callback(epsilon_theta, mean, old_x_T,  x_T, T):
            step = (st.session_state["steps"] - T - 1).item()
            log_step = st.session_state["steps"] // 10 
            if step % log_step == 0 :
                cols[step // log_step].image(x_T[0], caption=f"Step {step}", use_column_width = True) 
            if T == 1 :
                cols[-1].image(x_T[0], caption = f"Step {step}", use_column_width = True)
            if step == st.session_state["steps"] - 1 :
                cols[-1].image(x_T[0], caption = f"Step {step}", use_column_width = True) 
            arr.append(estimate_sigma(x_T[0]))
            progress_bar.progress(step / (st.session_state["steps"] - 1))
            data = pd.DataFrame({"Noise" : arr, "X" : list(range(len(arr)))})
            chart_placeholder.altair_chart(
                    alt.Chart(data).mark_line().encode(
                        y=alt.Y('Noise', title='Standard Deviation'),
                        x=alt.X('X', title='Timesteps')
                    ).properties(
                        title = f"Estimated Noise in the image",
                    ), 
                    use_container_width = True, 
                )
            if step % 5 == 0 :
                gif_placeholder.image(x_T[0], use_column_width = True, caption = f"Live Generations (Step {step})")
            if step == st.session_state["steps"] - 2 :
                gif_placeholder.image(x_T[0], use_column_width = True, caption = f"Live Generations (Step {step})")
        
        if st.session_state["steps"] < 1000 :
            ddpm.fast_generate(1, torch.tensor([names.index(selected_name)], device = ddpm.device), steps = st.session_state["steps"], lerp = st.session_state["lerp"], streamlit_callback = callback)
        else :
            ddpm.generate(1, torch.tensor([names.index(selected_name)], device = ddpm.device), streamlit_callback = callback)

        st.success("Generation complete!")

elif choice == "2D Diffusion":
    st.header("2D Diffusion")
    st.write("This is Option 2")

elif choice == "3D Diffusion":
    st.header("3D Diffusion")
    st.write("This is Option 3")
