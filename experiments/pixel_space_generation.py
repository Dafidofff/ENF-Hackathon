from absl import app
from tqdm import tqdm
from typing import List

import wandb
import matplotlib.pyplot as plt
import ml_collections
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import optax

# Custom imports
from datasets import get_dataloader
from models.downstream.unet import UNet


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False

    # Define the UNet model parameters
    config.unet = ml_collections.ConfigDict()
    config.unet.num_classes = 10
    config.unet.features = 64
    config.unet.layers = 4

    # Diffusion model config
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.use_cfg = False
    config.diffusion.cfg_val = 1.0
    config.diffusion.t_sampler = 'log-normal'
    config.diffusion.denoise_timesteps = 100

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "./data"
    config.dataset.name = "FIGURE"  # Choose between cifar10 and FIGURE
    config.dataset.num_signals_train = 1000  # Number of training signals (-1 for all)
    config.dataset.num_signals_test = 1000
    config.dataset.batch_size = 32
    config.dataset.num_workers = 8

    # Specific FIGURE dataset parameters (ONLY FOR FIGURE DATASET)
    config.dataset.figure_type = "FIGURE-Shape-B"  # Choose between FIGURE-Shape-B and FIGURE-Shape-CB
    config.dataset.swap_bias = False
    config.dataset.color_consistency = 0.9

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.lr = 5e-4
    config.train.num_epochs = 500
    config.train.log_interval = 5

    config.task = "pixel-generation"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_generations(generations: List[jnp.ndarray], batch_size: int = 32):
    """Plot CIFAR generations."""
    # min max normalise
    img_gen_min = jnp.min(generations)
    img_gen_max = jnp.max(generations)
    generations = (generations - img_gen_min) / (img_gen_max - img_gen_min)

    fig, axes = plt.subplots(8, batch_size // 8, figsize=(16, 16))
    for ax, img in zip(axes.flat, generations):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    return fig


# Diffusion model stuff
def get_img_t(img, eps, t):
    img_0 = eps
    img_1 = img
    t = jnp.clip(t, 0, 1 - 0.01)
    return (1 - t) * img_0 + t * img_1


def get_v(z, eps):
    z_0 = eps
    z_1 = z
    return z_1 - z_0


def main(_):
    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project="enf-hack", job_type="train", config=config.to_dict(), mode="online" if not config.debug else "dryrun")

    ##############################
    # Initializing the model
    ##############################

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloader(config.dataset)
    sample = next(iter(train_dloader))
    img_shape = sample[0].shape[1:]

    # Random key
    key = jax.random.PRNGKey(config.seed)

    # Define the model, num_classes should be set to the channel dimension since we predict the velocity field
    # for every color channel
    model = UNet(num_classes=img_shape[-1], features=config.unet.features, layers=config.unet.layers)

    # Initialize the model
    key, subkey = jax.random.split(key)
    dummy_img = jnp.ones((config.dataset.batch_size, *img_shape), dtype=jnp.float32)
    dummy_t = jnp.ones((config.dataset.batch_size, 1, 1, 1), dtype=jnp.float32)
    dummy_labels = jnp.ones((config.dataset.batch_size, 1), dtype=jnp.int32)
    model_params = model.init(subkey, dummy_img, dummy_t, dummy_labels)
    model_params_ema = model_params.copy()

    # Define optimizer for the ENF backbone
    optimizer = optax.adam(learning_rate=config.train.lr)
    model_opt_state = optimizer.init(model_params)

    @jax.jit
    def diffusion_train_step(img, labels, model_params, model_params_ema, opt_state, key):
        # Perform inner loop optimization to get latents
        key, time_key, noise_key = jax.random.split(key, 3)

        def diffusion_loss(model_params, img, labels):
            # Sample a t for training, log-normal
            t = jax.random.normal(time_key, (img.shape[0],))
            t = ((1 / (1 + jnp.exp(-t))))

            # Setup c_t and v_t
            t_full = t[:, None, None, None]
            eps_img = jax.random.normal(noise_key, img.shape)
            img_t = get_img_t(img, eps_img, t_full)
            v_img_t = get_v(img, eps_img)

            v_img_prime = model.apply(
                model_params,
                img_t,
                t,
            )
            loss = jnp.mean((v_img_prime - v_img_t) ** 2)
            return loss

        # Get gradients
        loss, grads = jax.value_and_grad(diffusion_loss)(model_params, img, labels)

        # Update diffusion model parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)

        # Update exponential moving average
        model_params_ema = jax.tree.map(
            lambda a, b: a * 0.999 + b * 0.001,
            model_params_ema,
            model_params,
        )

        return loss, model_params, model_params_ema, model_opt_state, key

    @jax.jit
    def denoise_step(img, t, model_params_ema):
        vc = model.apply(model_params_ema, img, t, labels)
        return vc

    @jax.jit
    def sample_diffusion_process(model_params_ema, key):
        # Sample a batch of images w/ normal distribution
        key, subkey = jax.random.split(key)
        noisy_img = jax.random.normal(subkey, (config.dataset.batch_size, *img_shape))

        # Perform the denoising process from x_0 (noise) to x_1 (data
        delta_t = 1.0 / config.diffusion.denoise_timesteps
        for ti in range(config.diffusion.denoise_timesteps):
            # Get the current time vector)
            t = ti / config.diffusion.denoise_timesteps
            t_vector = jnp.full((noisy_img.shape[0],), t)

            # Get the velocity field and update the noisy image
            v_img = denoise_step(noisy_img, t_vector, model_params_ema)
            noisy_img = noisy_img + v_img * delta_t

        return noisy_img

    # Training loop for the diffusion model
    glob_step = 0
    epoch_loss = []
    for epoch in tqdm(range(config.train.num_epochs), desc="Training epochs"):
        epoch_loss = []
        for i, batch in enumerate(train_dloader):
            img, labels = batch[0], batch[1]
            
            # Perform training step
            loss, model_params, model_params_ema, model_opt_state, key = diffusion_train_step(
                img, labels, model_params, model_params_ema, model_opt_state, key
            )

            epoch_loss.append(loss)
            glob_step += 1

            # Logging
            wandb.log({"diffusion-loss": loss}, step=glob_step)

        if epoch % config.train.log_interval == 0:
            # Sample diffusion process
            img_gen = sample_diffusion_process(model_params, key)
            fig = plot_generations(img_gen, config.dataset.batch_size)
            wandb.log({"diffusion-generations": fig}, step=glob_step, commit=False)
            plt.close('all')

            # Log the epoch loss
            wandb.log({"epoch-diffusion-loss": sum(epoch_loss) / len(epoch_loss)}, step=glob_step)
            epoch_loss = []

    run.finish()

if __name__ == "__main__":
    app.run(main)