import streamlit as st
from visual_diffusion_demo.visualise import *
from visual_diffusion_demo.model import model_loader
from visual_diffusion_demo.data import data_loader
from visual_diffusion_demo.beta_scheduler import beta_scheduler


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
        self.dim_steps = 31
        

if 'args' not in st.session_state:
    args = Args()
    st.session_state.args = args

st.header("Dataset Generation")
data = st.selectbox("Choose the dataset to train the model on", ['swissroll', 'circle', 'polygon', 'donut', 'spring', 'mobius', 'custom'])
numpoints = st.slider("Number of samples to generate for the dataset", 1000, 100000, 100000)

if (st.button("Generate the Dataset")):
    args = st.session_state.args
    dataset = data_loader(data = data, data_args = None, n = numpoints)
    fig = visualise_data(dataset, "Original Dataset", show = False)
    st.plotly_chart(fig, use_container_width = True)
    args.dataset_args(data, dataset, numpoints)
    st.write("Dataset generated successfully!")

st.header("Diffusion Parameters")
timesteps = st.slider("Number of timesteps", 10, 100, 10)
scheduler_type = st.selectbox("Choose the scheduler type", ['linear', 'quadratic'])
beta_min = st.slider("Minimum value of beta", 0.0001, 0.1, 0.001)
beta_max = st.slider("Maximum value of beta", 0.1, 1.0, 0.5)

if (st.button("Set Diffusion Parameters")):
    args = st.session_state.args
    args.diffusion_args(timesteps, scheduler_type, beta_min, beta_max)
    st.write("Diffusion parameters set successfully!")

st.header("Model Initalisation")
model_type = st.selectbox("Choose the model type to train on the data", ['Multi Layer Perceptron', 'Convolutional Neural Network'])
if (model_type == 'Multi Layer Perceptron'):
    hidden_dims = st.text_input("Hidden dimensions for the MLP model", '32-32')
if (model_type == 'Convolutional Neural Network'):
    hidden_dims = st.text_input("Channels for the CNN model", '32-32')
num_epochs = st.slider("Number of epochs", 0, 10000, 100)
lr = st.slider("Learning rate", 0.0001, 0.1, 0.001)
batch_size = 10000

if (st.button("Initialise the Model")):
    args = st.session_state.args
    if (args.data_init == False):
        st.error("Please generate the dataset first")
        st.stop()
    if (args.diff_init == False):
        st.error("Please set the diffusion parameters first")
        st.stop()
    args.model_init = True
    args.model_training_started = False
    dataset = st.session_state.args.data
    if (model_type == 'Multi Layer Perceptron'):
        model_type_ = 'mlp_diffusion'
    if (model_type == 'Convolutional Neural Network'):
        model_type_ = 'conv_diffusion'
    model = model_loader(model_type_, list(map(int, hidden_dims.split('-'))), dataset.shape[1], args.timesteps)
    args.model_args(model_type_, hidden_dims, num_epochs, lr, batch_size, model)
    st.write("Model initialised successfully!")

if (st.button("Train the Model")):
    args = st.session_state.args
    if (args.model_init == False):
        st.error("Please initialise the model first")
        st.stop()
    if (args.model_training_started == True):
        st.error("Model training had already been started! This might lead to inaccurate results. To ensure accurate results, please reinitalize the model and then start training.")
    args.inferset = False
    args.model_training_started = True
    progress_track = st.progress(0)
    model = args.model
    beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
    alpha_bar_ = beta_scheduler_.alpha_bar_schedule(args.timesteps)
    model.trainer(args.data, args.num_epochs, (args.n+args.batch_size-1)//args.batch_size, alpha_bar_, args.lr, 'cpu', 1, 'log.txt', progress_bar_callback = lambda x: progress_track.progress(x, text = f"Training Progress : {round(x*100)}%"))
    #args.model_path = f"{args.model.model_type}_diffusion_model_{args.dataname}_{args.n}_{args.timesteps}_{args.beta_min}_{args.beta_max}_{args.beta_scheduler}_{args.modeltype}.pt" # can also possibly include 
if (st.session_state.args.model_training_started == True):
    st.write("Model trained successfully!")

st.header("Model Inference")
numinferpoints = st.slider("Number of samples to generate for the dataset", 1000, 100000, 10000)
eta = st.slider("Value of eta", 0.0, 1.0, 1.0)

if (st.button("Set Inference Parameters")):
    args = st.session_state.args
    if (args.model_training_started == False):
        st.error("Please train the model first")
        st.stop()
    beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
    alpha_ = beta_scheduler_.alpha_schedule(args.timesteps)
    beta_ = beta_scheduler_.beta_schedule(args.timesteps)
    alpha_bar_ = beta_scheduler_.alpha_bar_schedule(args.timesteps)
    model = args.model
    dataset, timesteps_data, timesteps_drift = model.inferrer(args.n, args.n_dim, args.timesteps, eta, alpha_, alpha_bar_, beta_, 1, "cpu")
    st.write("Inference parameters set successfully!")
    fig = visualise_data(dataset, "Inferred Dataset", show = False)
    st.plotly_chart(fig, use_container_width = True)
    args.inference_args(eta, numinferpoints, dataset, timesteps_data, timesteps_drift, 5, 5)
    args.inferset = True
    

if (st.button("Visualise the Inference")):
    args = st.session_state.args
    if (args.inferset == False):
        st.error("Please set the inference parameters first")
        st.stop()
    fig = visualise_reverse_diffusion(args.timestepsdata, args.num_steps, show = False)
    st.plotly_chart(fig, use_container_width = True)
    st.write("Inference visualised successfully!")

if (st.button("Visualise the Drift")):
    args = st.session_state.args
    if (args.inferset == False):
        st.error("Please set the inference parameters first")
        st.stop()
    beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
    alpha_ = beta_scheduler_.alpha_schedule(args.timesteps)
    alpha_bar_ = beta_scheduler_.alpha_bar_schedule(args.timesteps)
    fig = visualise_reverse_drift(args, args.model, alpha_, alpha_bar_, show = False)
    st.plotly_chart(fig, use_container_width = True)
    st.write("Drift visualised successfully!")