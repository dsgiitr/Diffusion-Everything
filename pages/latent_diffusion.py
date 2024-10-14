import os
import time
import json 
import torch
from vae import *
import torchvision
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import noise_scheduler
import streamlit as st
from functools import partial


device = os.environ["DEVICE"]

st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_title="Latent Diffusion",
    page_icon = "content/favicon.png", 
)

st.markdown(f'''
    <style>
        {open("style.css").read()}
    </style>
''', unsafe_allow_html = True)

st.title("Latent Diffusion with MNIST")

st.header("VAE MNIST")
if 'initialized' not in st.session_state :
    st.session_state["vae"] = VAE(device = device).to(device)
    st.session_state["initialized"] = True

col3, col4 = st.columns([0.5, 0.5], gap = "large")

col3.markdown(open("content/vae.md").read(), unsafe_allow_html=True)

fig = col4.empty()

point_slider, epoch_slider = col4.columns([0.5, 0.5])
num_points = point_slider.slider("Num Points", min_value = 0, max_value = 1000, value = 500)
num_epochs = epoch_slider.slider("Epochs", min_value = 0, max_value = 10, value = 5)

button, progbar = col4.columns([0.2, 0.8])
chart3d = col4.container()

if 'latent' in st.session_state :
    fig.altair_chart(st.session_state["latent"], use_container_width = True)
if "chart3d" in st.session_state : 
    chart3d.plotly_chart(st.session_state["chart3d"])
if button.button("Train VAE"):
    vae = st.session_state["vae"]
    rloss = []
    kloss = []
    def callback(test_means, testy):
        st.session_state["means"] = test_means.numpy()
        st.session_state["labels"] = np.array(testy)
        x = test_means[:, 0].numpy()
        y = test_means[:, 1].numpy()
        labels = testy
        data = pd.DataFrame({
            'x': x,
            'y': y,
            'label': labels
        })
        chart = alt.Chart(data).mark_point().encode(
                x = alt.X('x:Q', scale=alt.Scale(domain=(-4, 4))),
                y = alt.X('y:Q', scale=alt.Scale(domain=(-4, 4))),
                color='label:N',
            tooltip=['x', 'y', 'label']
            ).properties(
                title='Latent Space', 
                height = 400, 
            )
        fig.altair_chart(chart.configure_title(
                anchor = 'middle'
            ), use_container_width = True) 
        st.session_state["latent"] = chart
    progbar = progbar.progress(0)
    vae.train(num_epochs, 128, num_points, streamlit_callback = callback, progress = progbar)
    col4.success("Training Complete")
    st.session_state["chart3d"] = generateFigure(
        data = st.session_state["means"],
        labels = st.session_state["labels"], 
    )
    chart3d.plotly_chart(st.session_state["chart3d"])

st.header("Latent Diffusion")
col6, col5 = st.columns([0.5, 0.5], gap = 'large')   

col6.markdown(open("content/latent_diffusion.md").read(), unsafe_allow_html=True)

scheduler = col5.selectbox("Choose forward noise scheduler", ("Cosine", "Linear"))
timesteps = col5.slider("Diffusion Timesteps", min_value = 0, max_value = 200, value = 100)
num_iters = col5.slider("Number of Iterations", min_value = 1000, max_value = 10000, value = 5000)
if scheduler == "Linear" :
    bs, be = col5.columns([0.5, 0.5])
    beta_start = bs.number_input("Beta Start", value = 1e-3)
    beta_end = be.number_input("Beta End", value = 1e-2)
col8, col9 = col5.columns([0.5, 0.5], gap = 'small')
if col8.button("Initialize Diffusion"): 
    with st.spinner("Encoding complete MNIST into latent space"):
        st.session_state["unet"] = UNet1D(timesteps = timesteps, device = device).to(device)
        if scheduler == "Cosine" :
            st.session_state["diffusion"] = Diffusion(
                unet = st.session_state["unet"], 
                vae = st.session_state["vae"], 
                scheduler = scheduler, 
                timesteps = timesteps, 
                device = device
            )
        else :
            st.session_state["diffusion"] = Diffusion(
                unet = st.session_state["unet"], 
                vae = st.session_state["vae"], 
                scheduler = scheduler, 
                timesteps = timesteps, 
                beta_start = beta_start, 
                beta_end = beta_end, 
                device = device
            )
            
diffusion_progbar = col8.empty()
if col8.button("Train 2D Latent Diffusion"):
    diffusion_train_progress = diffusion_progbar.progress(0)
    def callback(it, loss):
        diffusion_train_progress.progress(it / num_iters, text = f"Iter [{it}/{num_iters}]")
    st.session_state["diffusion"].train(
        num_iters = num_iters, 
        streamlit_callback = callback
    )

im = col9.empty()
st.session_state["label"] = col8.number_input("Number", min_value = 0, step = 1)
gen_images = col5.columns(11)
fig = col5.empty()
steps = torch.linspace(timesteps - 1, 0, 11).long().tolist()
if col8.button("Generate Image") :
    arrx = []
    arry = []
    diffusion_prog = col8.progress(0)  
    cnt = 0 
    def st_callback(t, img, pt):
        global cnt
        diffusion_prog.progress(1 - t.item()/(timesteps - 1), text = f"Step [{(timesteps - t.item())}/{timesteps}]")
        arrx.append(pt[0])
        arry.append(pt[1])
        time.sleep(0.05)
        data = pd.DataFrame({
            'x' : arrx,
            'y' : arry, 
        })
        point_chart = alt.Chart(data).mark_point().encode(
                x = alt.X('x', scale = alt.Scale(domain = (-4, 4))), 
                y = alt.Y('y', scale = alt.Scale(domain = (-4, 4)))
            ).properties(
                title = "Latent Space"
            )
        line_chart = alt.Chart(data).mark_line().encode(
                x = alt.X('x', scale = alt.Scale(domain = (-4, 4))), 
                y = alt.Y('y', scale = alt.Scale(domain = (-4, 4)))
            ).properties(
                title = "Latent Space"
            )
        fig.altair_chart(
            point_chart + line_chart + st.session_state["latent"],
            use_container_width = True, 
        )
        im.image(img, use_column_width = True, caption = "Live Generations")  
        if t.item() == steps[cnt] :
            gen_images[cnt].image(img, use_column_width = True, caption = f"Step {t.item()}")
            cnt += 1

    st.session_state["diffusion"].generate(st.session_state["label"], streamlit_callback = st_callback)


