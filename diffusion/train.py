import torch
import torch.nn.functional as F
from torch.optim import Adam


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_loss(model, x_0, t, makeup, eps=1e-5):
    x_noisy, noise = forward_diffusion_sample(x_0, t, eps)
    noise_pred = model(x_noisy, t, makeup)
    return F.mse_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t, model, makeup):
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

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, makeup, num_images=1):
    # Sample noise
    img = torch.randn((num_images, 3, 256, 256)).to(device)
    for i in range(0, T)[::-1]:
        t = torch.full((num_images,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model, makeup)
    return img


def train(model, dataloader, epochs, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for step, (face, makeup, _) in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (face.shape[0],)).long().to(device)
            loss = get_loss(model, face, t, makeup)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

# Assuming you have your dataloader ready
# train(model, dataloader, epochs=100)