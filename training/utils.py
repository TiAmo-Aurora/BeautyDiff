@torch.no_grad()
def sample(model, makeup, device, image_size=256, batch_size=16, channels=3):
    # Sample noise
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)

    # Sampling loop
    for i in tqdm(reversed(range(T)), desc='sampling loop time step', total=T):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model, makeup)

    return img