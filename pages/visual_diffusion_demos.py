import streamlit as st
from PIL import Image
import numpy as np
import torch 
from streamlit_drawable_canvas import st_canvas
import skimage as ski
from visual_diffusion_demo.visualise import *
from visual_diffusion_demo.model import model_loader, inferrer, trainer
from visual_diffusion_demo.data import data_loader
from visual_diffusion_demo.beta_scheduler import beta_scheduler
import altair as alt
import time
import noise_scheduler
import pandas as pd


st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed", 
    page_title = "2D/3D Diffusion Demos", 
    page_icon = "content/favicon.png", 
)
st.title("2D and 3D Diffusion Demonstration")
class Args:
    def __init__(self):
        self.dataname = None
        self.data = None
        self.n = None
        self.data_args = None
        self.datafile = None
        self.model = None
        self.hidden_dims = None
        self.num_epochs = None
        self.lr = None
        self.batch_size = None
        self.modeltype = None
        self.timesteps = None
        self.beta_scheduler = None
        self.beta_min = None
        self.beta_max = None
        self.data_init = False
        self.model_init = False
        self.diff_init = False
        self.model_training_started = False
        self.eta = None
        self.numinferpoints = None
        self.infdataset = None
        self.timestepsdata = None
        self.timestepsdrift = None
        self.num_steps = None
        self.drift_steps = None
        self.n_dim = None
        self.dim_min = None
        self.dim_max = None
        self.dim_steps = None
        self.inferset = False
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.epochs_trained = 0
        self.custom_points = None

    def dataset_args(self, dataname = None, dataset = None, numpoints = None, dataargs = None, datafile = None):
        self.data_init = True
        self.dataname = dataname
        self.data = dataset
        self.n = numpoints
        self.data_args = dataargs
        self.datafile = datafile
        self.n_dim = dataset.shape[1]
    
    def model_args(self, modeltype = None, hiddendims = None, numepochs = None, lr = None, batchsize = None, model = None):
        self.model_init = True
        self.modeltype = modeltype
        self.hidden_dims = hiddendims
        self.num_epochs = numepochs
        self.lr = lr
        self.batch_size = batchsize
        self.model = model
    
    def diffusion_args(self, timesteps = None, scheduler = None, beta_min = None, beta_max = None):
        self.diff_init = True
        self.timesteps = timesteps
        self.beta_scheduler = scheduler
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def inference_args(self, eta = None, numinferpoints = None, infdataset = None, timestepsdata = None, timestepsdrift = None, num_steps = None, drift_steps = None):
        self.eta = eta
        self.numinferpoints = numinferpoints
        self.infdataset = infdataset
        self.timestepsdata = timestepsdata
        self.timestepsdrift = timestepsdrift
        self.num_steps = num_steps
        self.drift_steps = drift_steps
        self.dim_min = -15
        self.dim_max = 15
        self.dim_steps = 15
        

if 'args' not in st.session_state:
    args = Args()
    st.session_state.args = args

st.header("Dataset Generation")

col1, col2 = st.columns([0.5, 0.5], gap = "large")

if 'datagraph' not in st.session_state :
    st.session_state["datagraph"] = col2.empty()
else :
    if 'fig' in st.session_state :
        st.session_state["datagraph"].plotly_chart(st.session_state["fig"], use_container_width = True)

def set_fig():
    if 'fig' in st.session_state :
        del st.session_state['fig']

data = col1.selectbox("Choose the dataset to train the model on", ['swissroll', 'circle', 'polygon', 'donut', 'spring', 'mobius', 'custom'], on_change = set_fig)
numpoints = col1.slider("Number of samples to generate for the dataset", 0, 100, 1000)
if (data == 'custom') and 'fig' not in st.session_state:
    custom_points = None
    stroke_width = col1.slider('Stroke width', 1, 50, 10)
    canvas_size = 501
    dataset = None 
    with st.session_state["datagraph"] : 
        dataset = st_canvas(update_streamlit = True, height = canvas_size, width = canvas_size, stroke_width = stroke_width)
    if dataset is not None and dataset.image_data is not None:
        img_data = dataset.image_data
        im = img_data[:,:,3]
        curpoints = np.count_nonzero(im)
        if curpoints > 0:
            resize_ratio = (numpoints+curpoints-1)//curpoints
            im = ski.transform.resize(im, (resize_ratio*im.shape[0], resize_ratio*im.shape[1]), anti_aliasing = True)
        custom_points = []
        new_canvas_size = im.shape[0]
        for i in range(new_canvas_size):
            for j in range(new_canvas_size):
                if im[i, j]> 0:
                    custom_points.append([j-(new_canvas_size-1)/2,(new_canvas_size-1)/2-i])
        custom_points = np.array(custom_points)
        if (custom_points is not None and custom_points.size > 0 and np.max(np.abs(custom_points)) > 0):
            custom_points = custom_points/np.max(np.abs(custom_points))*15
        st.session_state.args.custom_points = custom_points

if (col1.button("Generate the Dataset")):
    custom_points = st.session_state.args.custom_points
    if (custom_points is None or custom_points.shape[0] == 1) and data == 'custom':
        st.error("Please draw on the canvas first")
        st.stop()
    args = st.session_state.args
    dataset = data_loader(data = data, data_args = None, n = numpoints, datafile = custom_points)
    fig = visualise_data(dataset, "Original Dataset", show = False)
    st.session_state["fig"] = fig 
    st.session_state["datagraph"].plotly_chart(st.session_state["fig"], use_container_width = True)
    args.dataset_args(data, dataset, numpoints)
    col1.success("Dataset generated successfully!")

col3, col4 = st.columns([0.5, 0.5], gap = 'large')
col3.header("Diffusion Parameters")
timesteps = col3.slider("Number of timesteps", 10, 100, 10)
scheduler_type = col3.selectbox("Choose the scheduler type", ['linear', 'quadratic', 'cosine'])

beta_max = None 
beta_min = None
if scheduler_type != 'cosine':
    beta_min = col3.slider("Minimum value of beta", min_value = 0.0001, max_value = 0.1, step = 0.001, value = 0.0001)
    beta_max = col3.slider("Maximum value of beta", min_value = 0.0001, max_value = 0.8, step = 0.001, value = 0.0100)

if (col3.button("Set Diffusion Parameters")):
    args = st.session_state.args
    if scheduler_type != 'cosine':
        if beta_min > beta_max :
            col3.error("Make sure beta_min < beta_max")
            st.stop()
    args.diffusion_args(timesteps, scheduler_type, beta_min, beta_max)
    col3.success("Diffusion parameters set successfully!")

col4.header("Model Initalisation")
model_type = col4.selectbox("Choose the model type to train on the data", ['Multi Layer Perceptron', 'Convolutional Neural Network'])
if (model_type == 'Multi Layer Perceptron'):
    hidden_dims = col4.text_input("Hidden dimensions for the MLP model", '32-32')
if (model_type == 'Convolutional Neural Network'):
    hidden_dims = col4.text_input("Channels for the CNN model", '32-32')
num_epochs = col4.slider("Number of epochs", 0, 1000, 100)
lr = col4.slider("Learning rate", min_value = 0.0001, max_value = 0.1, step = 0.001, value = 0.001)
batch_size = 10000

if (col4.button("Initialise the Model")):
    args = st.session_state.args
    if (args.data_init == False):
        col4.error("Please generate the dataset first")
        st.stop()
    if (args.diff_init == False):
        col4.error("Please set the diffusion parameters first")
        st.stop()
    args.num_epochs = num_epochs
    args.model_init = True
    args.model_training_started = False
    dataset = st.session_state.args.data
    if (model_type == 'Multi Layer Perceptron'):
        model_type_ = 'mlp_diffusion'
    if (model_type == 'Convolutional Neural Network'):
        model_type_ = 'conv_diffusion'
    model = model_loader(model_type_, list(map(int, hidden_dims.split('-'))), dataset.shape[1], args.timesteps)
    model = model.to(args.device)
    model.device = args.device
    args.epochs_trained = 0
    args.model_args(model_type_, hidden_dims, num_epochs, lr, batch_size, model)
    col4.success("Model initialised successfully!")

st.header("Training")
if (st.button("Train the Model")):
    args = st.session_state.args
    if (args.model_init == False):
        st.error("Please initialise the model first")
        st.stop()
    if (args.model_training_started == True):
        st.warning(f"You have already trained the model for {args.epochs_trained} epochs, this will now train the model for {args.num_epochs} more epochs")
    args.inferset = False
    args.model_training_started = True
    progress_track = st.progress(0)
    model = args.model 
    if args.beta_scheduler == "cosine" :
        model.ns = noise_scheduler.Cosine(timesteps = args.timesteps, device = args.device)
    elif args.beta_scheduler == "linear" :
        model.ns = noise_scheduler.Linear(beta_start = args.beta_min, beta_end = args.beta_max, device = args.device, timesteps = args.timesteps)
    trainer(model, args.data, args.num_epochs, (args.n+args.batch_size-1)//args.batch_size, args.lr, args.device, 1, 'log.txt', progress_bar_callback = lambda x: progress_track.progress(x, text = f"Training Progress : {round(x*100)}%"))
    args.epochs_trained += args.num_epochs
if (st.session_state.args.model_training_started == True):
    st.success("Training Complete !")

col5, col6 = st.columns([0.5, 0.5], gap = 'large')

col5.header("Model Inference")
numinferpoints = col5.slider("Number of samples to generate for the dataset", 1000, 100000, 10000)
eta = col5.slider("Value of eta", 0.0, 1.0, 1.0)
random_image = col6.empty()
if (col5.button("Generate")):
    args = st.session_state.args
    if (args.model_training_started == False):
        st.error("Please train the model first")
        st.stop()
    model = args.model  
    images = st.container()
    reverse_cols = images.columns(11)
    steps = torch.linspace(0, args.timesteps, 11).int().tolist()
    cnt = 0 
    def callback(x_t, t):
        global cnt 
        x_t = x_t.cpu().numpy()
        idx = args.timesteps - t 
        datapoints = pd.DataFrame({
            "x" : x_t[:, 0],
            "y" : x_t[:, 1]
        })
        chart = alt.Chart(datapoints).mark_point().encode(
                    x = alt.X('x', scale=alt.Scale(domain=(-20, 20))),
                    y = alt.X('y', scale=alt.Scale(domain=(-20, 20))),
                ).properties(
                    title = "Reverse Diffusion"
                ).configure_title(
                    anchor = 'middle'
                )
        if steps[cnt] == idx :
            pass
        time.sleep(0.1)
        random_image.altair_chart(chart)
    dataset, timesteps_data, timesteps_drift = inferrer(model, numinferpoints, args.n_dim, args.timesteps, eta, 1, args.device, streamlit_callback = callback)
    col5.success("Data generated !")
    fig = visualise_data(dataset, "Inferred Dataset", show = False)
    args.inference_args(eta, numinferpoints, dataset, timesteps_data, timesteps_drift, 5, 5)
    args.inferset = True

col7, col8 = st.columns([0.5, 0.5], gap = 'large')
drifts = st.container()
if (col7.button("Visualise the Drift")):
    args = st.session_state.args
    drift_cols = drifts.columns(args.drift_steps)
    if (args.inferset == False):
        st.error("Please set the inference parameters first")
        st.stop()
    beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
    alpha_ = beta_scheduler_.alpha_schedule(args.timesteps)
    alpha_bar_ = beta_scheduler_.alpha_bar_schedule(args.timesteps)
    cnt = 0
    def callback(fig, t):
        global cnt 
        time.sleep(0.5)
        fig.write_image("chart.png")
        drift_cols[cnt].image(np.array(Image.open("chart.png")))
        cnt += 1
    fig = visualise_reverse_drift(args, args.model, alpha_, alpha_bar_, show = False, device = args.device, streamlit_callback = callback)
    st.success("Drift visualised successfully!")