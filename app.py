import streamlit as st


st.set_page_config(layout="wide")


st.title("Diffusion Everything")

st.write("Welcome to Diffusion Everything! Please select a page to access the corresponding demonstration.")
menu = ["Title Page", "DDPM - Class Conditioned", "2D and 3D Diffusion Demonstration", "Option 3"]
choice = st.selectbox("Menu", menu)

if choice == "DDPM - Class Conditioned":
    st.switch_page("pages/class_conditioned_ddpm_demo.py")

elif choice == "2D and 3D Diffusion Demonstration":
    st.switch_page("pages/visual_diffusion_demos.py")
