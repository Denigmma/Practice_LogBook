import torch

def get_multi_boundary_timesteps(timesteps, num_boundaries=4, num_timesteps=1000):
    step = num_timesteps // num_boundaries

    boundaries = torch.arange(0, num_timesteps + 1, step, device=timesteps.device)

    boundary_indices = torch.bucketize(timesteps, boundaries, right=True) - 1

    boundary_indices = boundary_indices.clamp(min=0, max=num_boundaries -1)

    boundary_timesteps = boundaries[boundary_indices]

    return boundary_timesteps
