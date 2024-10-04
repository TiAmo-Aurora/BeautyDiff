import torch
from torch.optim import Adam
from torch.nn import functional as F
from models.perceptual_loss import PerceptualLoss
from tqdm import tqdm
import numpy as np

# Define constants for the diffusion process
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def get_loss(model, x_0, t, makeup, perceptual_loss_fn, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, makeup)
    mse_loss = F.mse_loss(noise, noise_pred)
    perceptual_loss = perceptual_loss_fn(noise_pred, noise)
    return mse_loss + perceptual_loss


@torch.no_grad()
def sample_timestep(x, t, model, makeup):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, makeup) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def train(model, dataloader, epochs, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    perceptual_loss_fn = PerceptualLoss().to(device)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader)
        for step, (face, makeup) in enumerate(pbar):
            face = face.to(device)
            makeup = makeup.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, T, (face.shape[0],), device=device).long()
            loss = get_loss(model, face, t, makeup, perceptual_loss_fn, device)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item(), "epoch": epoch})

        # Save model
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')

        print(f"Epoch {epoch} | Loss: {loss.item():.5f}")
