First we'll be training a simple unconditional VAE (Variational AutoEncoder) on MNIST dataset. For those who aren't familiar, with Autoencoders we map the input to a fixed vector but in Varitational Autoencoders we map the input $p(x)$ to a latent distribution $p(z)$ and our goal is to maximize the likelihood of generating real data samples. For training we use a surrogative objective of maximizing ELBO (Evidence Lower Bound) which itself is a lower bound of log likelihood. 

$$
ELBO = \mathbb{E}_{z\sim q\phi(z|x)} logp_\theta(x|z) + D_{KL}(q_\phi(z|x)|p_\theta(z))
$$

Intuitively we are minizing mean squared error at the same time regularizing the loss by adding a KL Divergence term. 
Here is the code for the training loop : 
```python
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        x = x.to(self.device)
        recon, mean, log_var = self(x)
        recon_loss = F.mse_loss(recon, x, reduction = 'sum')
        kl_loss = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) 
        loss = recon_loss + kl_loss (negative ELBO)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
To give better representation of the latent space, we applied 2D Kernel Density estimation which estimates the probability density function of a dataset by smoothing data points using a kernel function. Each color correnponds to latent distribution of a specific number.  

```python
unique_labels = np.unique(labels)
xmin, ymin = data.min(axis=0) - 1
xmax, ymax = data.max(axis=0) + 1
x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([x.ravel(), y.ravel()])
fig = go.Figure()
colors = [...]
for i, label in enumerate(unique_labels):
    class_data = data[labels == label].T
    kde = gaussian_kde(class_data)
    z = np.reshape(kde(positions).T, x.shape)
    fig.add_trace(
        go.Surface(z=z, x=x, y=y, 
            colorscale=[[0, colors[i]], [1, colors[i]]]
        )
    )
```
