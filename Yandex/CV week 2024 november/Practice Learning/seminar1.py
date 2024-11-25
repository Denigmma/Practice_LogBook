import torch
import torchvision
import time
import math

import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


x, y = next(iter(train_dataloader))

plt.figure(figsize=(14, 4))
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.axis('off')
plt.show()

alpha_t = .1

x_t = (1 - alpha_t) * x[0] + alpha_t * torch.randn_like(x[0])
plt.imshow(x_t[0], cmap='Grays')
plt.axis('off')
plt.show()


def add_noise(x, alpha_t):
    noise = torch.randn_like(x)
    alpha_t = alpha_t.view(-1, 1, 1, 1)  # <- squeeze all dimentions but one into the first dimention
    return x * (1 - alpha_t) + noise * alpha_t



single_x = x[0]
x_to_show = single_x.repeat(x.shape[0], 1, 1, 1)

fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x_to_show)[0], cmap='Greys')

# Adding noise
alpha = torch.linspace(0, 1, x.shape[0])
noised_x = add_noise(x_to_show, alpha)

# Plotting the noised version
axs[1].set_title('Corrupted data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys');


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            # Use padding 2 to make 28x28 -> 30x30, otherwise we would get a RuntimeError
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Continue using padding to preserve resolution
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x

net = BasicUNet()
x = torch.randn(8, 1, 28, 28)
net(x).shape

sum([p.numel() for p in net.parameters()])

# Dataloader (choose BS to fit into your GPU, up to 8192 fits into 12Gb VRAM GPUs e.g. 1080 Ti)
batch_size = 128
n_epochs = 3
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the network
net = BasicUNet()
net.to(device)

# Our loss function (`L_simple` in Lecture 1)
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
basic_unet_losses = []

# The training loop
for epoch in range(n_epochs):

    epoch_start_time = time.time()
    for x, _ in train_dataloader:
        x = x.to(device)
        alpha_t = torch.rand(x.shape[0]).to(device)    # Pick random noise amounts
        noisy_x = add_noise(x, alpha_t)                # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x)

        # Calculate the loss
        # We make the model to predict `x_0` for now
        # We will see other forms of problem formulation later
        loss = loss_fn(pred, x)

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        basic_unet_losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(basic_unet_losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    # Calculate and print the time per epoch of training
    epoch_end_time = time.time()
    print(f'This epoch took {round(epoch_end_time - epoch_start_time, 1)} sec to finish')

# View the loss curve
plt.plot(basic_unet_losses)
plt.ylim(0, 0.1)



# 1. Load the data
x, y = next(iter(train_dataloader))

# Let us look at a single image so we see the difference more clearly
n_imgs_to_show = 8
x_to_show = x[0].repeat(n_imgs_to_show, 1, 1, 1)

# 2. Corrupt it with our noising process
alpha = torch.linspace(0, 1, x_to_show.shape[0])
noised_x = add_noise(x_to_show, alpha)

# 3. Get model predictions for all levels of noise
with torch.no_grad():
  preds = net(noised_x.to(device)).detach().cpu()

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x_to_show)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Corrupted data')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys');



# 1. Sample a random noise
x = torch.randn(8, 1, 28, 28).to(device)

# Store intermediate results for visualization
step_history = [x.detach().cpu()]
pred_output_history = []

# Repeat for a `num_of_sampling_steps` times
num_of_sampling_steps = 5
for i in range(num_of_sampling_steps):
    # 2. Get an estimate of generated image
    with torch.no_grad():
        pred = net(x)

    pred_output_history.append(pred.detach().cpu())  # Store model output for plotting

    mix_factor = 1 / (num_of_sampling_steps - i)  # How much we move towards the prediction
    x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
    step_history.append(x.detach().cpu())  # Store step for plotting

fig, axs = plt.subplots(num_of_sampling_steps, 2, figsize=(14, 6), sharex=True)
axs[0, 0].set_title('x (model input)')
axs[0, 1].set_title('model prediction')
for i in range(num_of_sampling_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        # We use half the dimension because we'll generate sin and cos for each dimension, which will then be concatenated
        half_dim = self.dim // 2

        # This creates a constant used to space out the frequency bands.
        # The constant 10000 comes from the original Transformer paper and works well in practice.
        # Dividing by `(half_dim - 1)` ensures that the frequencies span from 1 to 10000 evenly in log-space
        embeddings = math.log(10000) / (half_dim - 1)

        # This creates a tensor [0, 1, ..., half_dim-1].
        # Multiplying by -embeddings and then applying exp creates a tensor of decreasing values from 1 to 1/10000.
        # This generates the frequency bands for the sinusoidal functions
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # This multiplies each timestep by all frequency bands and adds a dimension to embeddings, making it a row vector.
        # The result is a 2D tensor where each row corresponds to a timestep, and each column to a frequency
        embeddings = time[:, None] * embeddings[None, :]

        # This applies sin and cos functions to the embeddings.
        # The results are concatenated along the last dimension.
        # This gives the final embedding where odd indices are sin and even indices are cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size, padding, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size, padding=padding)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm_1 = nn.GroupNorm(32, out_channels)
        self.batch_norm_2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.batch_norm_1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))

        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]

        # Add time channel
        h = h + time_emb

        # Second Conv
        h = self.batch_norm_2(self.relu(self.conv2(h)))

        # Down or Upsample
        return self.transform(h)


class TimeUNet(nn.Module):
    """A minimal UNet implementation with timestep conditioning."""

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=28):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)

        # Downsample
        self.downs = nn.ModuleList([
            Block(32, 64, time_emb_dim, kernel_size=5, padding=2),
            Block(64, 64, time_emb_dim, kernel_size=5, padding=2)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(64, 64, time_emb_dim, kernel_size=5, padding=2, up=True),
            Block(64, 32, time_emb_dim, kernel_size=5, padding=2, up=True)
        ])

        self.output = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


net = TimeUNet()
sum([p.numel() for p in net.parameters()])

batch_size = 128
n_epochs = 3
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the network
net = TimeUNet()
net.to(device)

# Our loss function (L_simple)
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
time_unet_losses = []

# The training loop
for epoch in range(n_epochs):

    epoch_start_time = time.time()
    for x, _ in train_dataloader:
        x = x.to(device)
        t = torch.rand(x.shape[0]).to(device)          # Pick random noise amounts
        noisy_x = add_noise(x, t)                      # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x, t)

        # Calculate the loss
        # We make the model to predict x_0 for now
        # We will see other forms of problem formulation later
        loss = loss_fn(pred, x)

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        time_unet_losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(time_unet_losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    # Calculate and print the time per epoch of training
    epoch_end_time = time.time()
    print(f'This epoch took {round(epoch_end_time - epoch_start_time, 1)} sec to finish')


from scipy.ndimage import gaussian_filter1d

# Function to apply moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to apply Gaussian smoothing
def gaussian_smooth(data, sigma=2):
    return gaussian_filter1d(data, sigma)

# Plot losses and some samples
fig, axs = plt.subplots(1, 1, figsize=(12, 5))

# Smoothing parameters
window_size = 20  # for moving average
sigma = 2  # for Gaussian smoothing

# Losses
basic_line = axs.plot(gaussian_smooth(basic_unet_losses), color='b', label='Basic UNet', linewidth=1)
time_line = axs.plot(gaussian_smooth(time_unet_losses), color='r', label='Time UNet', linewidth=1)

axs.set_ylim(0, 0.1)
axs.set_title('Loss over time')
axs.legend()

# Add grid for better readability
axs.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# 1. Sample a random noise
x = torch.randn(8, 1, 28, 28).to(device)

# Store intermediate results for visualization
step_history = [x.detach().cpu()]
pred_output_history = []

# Repeat for a `num_of_sampling_steps` times
num_of_sampling_steps = 5
for i in range(num_of_sampling_steps):
    # 2. Get an estimate of generated image
    t = torch.tensor([1 / (num_of_sampling_steps - i)]).cuda()
    with torch.no_grad():
        pred = net(x, t)

    pred_output_history.append(pred.detach().cpu())  # Store model output for plotting

    mix_factor = 1 / (num_of_sampling_steps - i)  # How much we move towards the prediction
    x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
    step_history.append(x.detach().cpu())  # Store step for plotting

fig, axs = plt.subplots(num_of_sampling_steps, 2, figsize=(14, 6), sharex=True)
axs[0, 0].set_title('x (model input)')
axs[0, 1].set_title('model prediction')
for i in range(num_of_sampling_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')


from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")


fig, axs = plt.subplots(2, 1, figsize=(14, 5))
xb, _ = next(iter(train_dataloader))
xb = xb.to(device)[0].repeat(8, 1, 1, 1)
xb = xb * 2. - 1.            # Map to (-1, 1)

# Show clean inputs
axs[0].imshow(torchvision.utils.make_grid(xb[:8])[0].detach().cpu(), cmap='Greys')
axs[0].set_title('Clean X')

# Add noise with scheduler
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)

# Show noisy version (with and without clipping)
axs[1].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu().clip(-1, 1),  cmap='Greys')
axs[1].set_title('Noisy X (clipped to (-1, 1)')
plt.show()


cosine_noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
plt.plot(cosine_noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - cosine_noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")

# Add noise with scheduler
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb[:8])
cosine_noisy_xb = cosine_noise_scheduler.add_noise(xb[0].repeat(8, 1, 1, 1), noise, timesteps)

fig, axs = plt.subplots(2, 1, figsize=(14, 5))
axs[0].imshow(torchvision.utils.make_grid(noisy_xb)[0].detach().cpu().clip(-1, 1),  cmap='Greys')
axs[0].set_title('Linear Schedule: Noisy X (clipped to (-1, 1)')
axs[1].imshow(torchvision.utils.make_grid(cosine_noisy_xb)[0].detach().cpu().clip(-1, 1),  cmap='Greys')
axs[1].set_title('Cosine Schedule: Noisy X (clipped to (-1, 1)')

from diffusers import UNet2DModel

net = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 64), # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",               # a regular ResNet downsampling block
        "AttnDownBlock2D",           # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",             # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",                 # a regular ResNet upsampling block
      ),
)

print(net)

net.to(device)

sum([p.numel() for p in net.parameters()])

n_epochs = 3
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Our loss finction
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
diffusers_unet_losses = []

# Forward scheduler
cosine_noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# The training loop
for epoch in range(n_epochs):

    epoch_start_time = time.time()
    for x, _ in train_dataloader:
        x = x.to(device)

        # Explicitly sample noise here
        noise = torch.randn_like(x).to(device)

        # Sample a random timestep for each image
        timestep = torch.randint(0, cosine_noise_scheduler.config.num_train_timesteps, (x.shape[0],),
                                 device=x.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_x = cosine_noise_scheduler.add_noise(x, noise, timestep)

        # Get the model prediction
        noise_pred = net(noisy_x, timestep).sample

        # Calculate the loss
        loss = loss_fn(noise_pred, noise)

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        diffusers_unet_losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(diffusers_unet_losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    # Calculate and print the time per epoch of training
    epoch_end_time = time.time()
    print(f'This epoch took {round(epoch_end_time - epoch_start_time, 1)} sec to finish')


from scipy.ndimage import gaussian_filter1d

# Function to apply moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to apply Gaussian smoothing
def gaussian_smooth(data, sigma=2):
    return gaussian_filter1d(data, sigma)

# Plot losses and some samples
fig, axs = plt.subplots(1, 1, figsize=(12, 5))

# Smoothing parameters
window_size = 20  # for moving average
sigma = 2  # for Gaussian smoothing

# Losses
diff_line = axs.plot(gaussian_smooth(diffusers_unet_losses), color='orange', label='Diffusers UNet', linewidth=1)
basic_line = axs.plot(gaussian_smooth(basic_unet_losses), color='b', label='Basic UNet', linewidth=1)
time_line = axs.plot(gaussian_smooth(time_unet_losses), color='r', label='Time UNet', linewidth=1)

axs.set_ylim(0, 0.1)
axs.set_title('Loss over time \n Note that direct comparison is not really correct')
axs.legend()

# Add grid for better readability
axs.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

import tqdm

noisy_sample = torch.randn_like(x).to(device)
sample = noisy_sample

scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2')
scheduler.set_timesteps(num_inference_steps=200)

samples_to_show = []
for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = net(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 10 == 0:
        samples_to_show.append(sample)


plt.figure(figsize=(14, 8))
for i, s in enumerate(samples_to_show):
    plt.subplot(4, 5, i + 1)
    plt.imshow(s.detach().cpu().numpy()[0, 0], cmap='Grays')
    plt.title(f'Step {10 + i * 10}')
    plt.axis('off')

plt.show()

plt.figure(figsize=(14, 8))
plt.imshow(torchvision.utils.make_grid(samples_to_show[-1].detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
plt.axis('off')
plt.show()






















