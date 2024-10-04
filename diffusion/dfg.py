import torch


def diffusion_free_guidance(model, face, makeup, num_inference_steps, guidance_scale=7.5):
    # Create a classifier-free guidance sampling function
    def model_fn(x_t, t_batch, makeup):
        # Get model output for conditional generation
        model_output = model(x_t, t_batch, makeup)

        # Get model output for unconditional generation
        unconditional_output = model(x_t, t_batch, torch.zeros_like(makeup))

        # Perform guidance
        return unconditional_output + guidance_scale * (model_output - unconditional_output)

    # Start from random noise
    x_t = torch.randn_like(face)

    # Iteratively denoise the image
    for t in reversed(range(num_inference_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # Estimate noise using the model with guidance
        estimated_noise = model_fn(x_t, t_batch, makeup)

        # Update x_t using the estimated noise
        x_t = sample_timestep(x_t, t_batch, lambda *args: estimated_noise, makeup)

    return x_t

# Usage example:
# face = ... # input face image
# makeup = ... # reference makeup image
# num_inference_steps = 50
# result = diffusion_free_guidance(model, face, makeup, num_inference_steps)