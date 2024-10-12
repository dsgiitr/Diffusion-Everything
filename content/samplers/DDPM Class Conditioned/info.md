###### Reverse Transition
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_{\theta}(x_t, t)) + \sigma_t z
$$  

The above equation can be derived from the fact that $q(x_{t-1}|{x_t})$ is eaxctly tracable given we have $x_0$. The model $\epsilon_{\theta}(x_t, t)$ outputs the estimated noise that was added at $t_0$ which can be used to calculate $x_0$ by reversing forward transition equation.   

To calculate $q(x_{t-1}|x_t, x_0)$ we can use simple bayes rule :  
$$
q(x_{t-1}|x_t, x_0) = \frac{q(x_t | x_{t-1},  x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)}
$$

$q(x_t | x_{t-1},  x_0)$ is the probability density function of the distribution denoted by [Eq1], similarly $q(x_{t-1} | x_0)$ and $q(x_t | x_0)$ can be denoted by [Eq2]. Since all the distributions are gaussian, and product of two gaussians is a gaussian, we obtain the resulting distrubiton as gaussian with mean $\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_{\theta}(x_t, t))$ and $\sigma_t$ standard deviation, which can be used to sample the previous timestep image. 

Now, for conditioning the model with some label $y$ trainable embeddings are injected during the denoising process in every pass of UNet along with timestep embedding. So we are actually calculating $q(x_{t-1} | x_t, x_0, y)$ 


