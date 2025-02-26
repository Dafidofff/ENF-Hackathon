import os 
import math
import logging
from pathlib import Path

import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import wandb

# Custom datasets
from datasets import get_dataloader
from models.enf.model import EquivariantNeuralField
from models.enf.bi_invariants import get_bi_invariant


os.environ['JAX_PLATFORM_NAME'] = 'gpu' 

def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False

    # Define the ENF model parameters
    config.enf = ml_collections.ConfigDict()
    config.enf.num_in = 2           # Images are 2D
    config.enf.num_out = 3          # RGB images = 3 channels, grayscale = 1
    config.enf.num_hidden = 128
    config.enf.num_heads = 3
    config.enf.att_dim = 128
    config.enf.num_latents = 4
    config.enf.latent_dim = 32
    config.enf.freq_multiplier_query = 1.0 #15.0
    config.enf.freq_multiplier_value = 2.0 #25.0
    config.enf.k_nearest = 4
    config.enf.bi_invariant = "translation"         # Choose between translation and roto_translation_2d 
                                                    # or implement your own bi-invariant!

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "./data"
    config.dataset.name = "FIGURE"                  # For this hackthon choose between cifar10 and FIGURE
    config.dataset.num_signals_train = 1000
    config.dataset.num_signals_test = 1000
    config.dataset.batch_size = 32
    config.dataset.num_workers = 8

    # Specific FIGURE dataset parameters (ONLY FOR FIGURE DATASET)
    config.dataset.figure_type = "FIGURE-Shape-B"             # Choose between FIGURE-Shape-B and FIGURE-Shape-CB
    config.dataset.swap_bias = False                          # Mainly interesting for down-stream tasks
    config.dataset.color_consistency = 0.9
    
    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr_c = 15.
    config.optim.inner_lr_p = 0.
    config.optim.inner_lr_g = 0.
    config.optim.inner_steps = 3

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.num_epochs = 5000
    config.train.log_interval = 100
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.checkpoint_path = f"./checkpoints/{config.dataset.name}/"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


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
    key = jax.random.PRNGKey(55)

    # Create coordinate grid
    x = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, img_shape[0]), jnp.linspace(-1, 1, img_shape[1])), axis=-1)
    x = jnp.reshape(x, (-1, 2))
    x = jnp.repeat(x[None, ...], config.dataset.batch_size, axis=0)

    # Define the model
    bi_inv = get_bi_invariant(config.enf.bi_invariant)
    model = EquivariantNeuralField(
        num_hidden=config.enf.num_hidden,
        att_dim=config.enf.att_dim,
        num_heads=config.enf.num_heads,
        num_out=config.enf.num_out,
        emb_freq_q=config.enf.freq_multiplier_query,
        emb_freq_v=config.enf.freq_multiplier_value,
        nearest_k=config.enf.k_nearest,
        bi_invariant=bi_inv,
    )


    ######################################################################################################
    ### NOTE: With jax we first need to initialise the model parameters by passing inputs through        #
    ### the model s.t. the model can infer the shapes of the parameters for jit compilation. This is     #
    ### done by passing dummy latents through the model.                                                 #
    ######################################################################################################

    # Create dummy latents for model init
    d_p = jnp.ones((config.dataset.batch_size, config.enf.num_latents, 2))                        # poses
    d_c = jnp.ones((config.dataset.batch_size, config.enf.num_latents, config.enf.latent_dim))    # context vectors
    d_g = jnp.ones((config.dataset.batch_size, config.enf.num_latents, 1))                        # gaussian window parameter

    # Init the model
    enf_params = model.init(key, x, d_p, d_c, d_g)

    # Define optimizer for the ENF backbone
    enf_optimizer = optax.adam(learning_rate=config.optim.lr_enf)
    enf_opt_state = enf_optimizer.init(enf_params)

    # Define checkpointing
    checkpoint_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=1,
    )
    checkpoint_manager = ocp.CheckpointManager(
        directory=Path(config.checkpoint_path + f'/{run.name}').absolute(),
        options=checkpoint_options,
        item_handlers={
            'state': ocp.StandardCheckpointHandler(),
            'config': ocp.JsonCheckpointHandler(),
        },
        item_names=['state', 'config']
    )

    ##############################
    # Training logic
    ##############################
    @jax.jit
    def inner_loop(enf_params, x_i, y_i, key):
        # Sample poses
        if config.enf.num_latents == 1:
            poses = jnp.zeros((1, config.enf.num_latents, 2))
        else:
            lims = 1 - 1 / math.sqrt(config.enf.num_latents)
            poses = jnp.stack(jnp.meshgrid(jnp.linspace(-lims, lims, int(math.sqrt(config.enf.num_latents))),
                                           jnp.linspace(-lims, lims, int(math.sqrt(config.enf.num_latents)))), axis=-1)
            poses = jnp.reshape(poses, (1, -1, 2))
        poses = poses.repeat(config.dataset.batch_size, axis=0)

        # Add some noise to the poses
        poses = poses + jax.random.normal(key, poses.shape) * 0.1 / jnp.sqrt(config.enf.num_latents)

        # Initialize values for the poses, note that these depend on the bi-invariant, context and window
        c = jnp.ones((x_i.shape[0], config.enf.num_latents, config.enf.latent_dim)) / config.enf.latent_dim  # context vectors
        g = jnp.ones((x_i.shape[0], config.enf.num_latents, 1)) * 2 / jnp.sqrt(config.enf.num_latents)  # gaussian window parameter

        def mse_loss(z, x_i, y_i):
            out = model.apply(enf_params, x_i, *z)
            return jnp.sum(jnp.mean((out - y_i) ** 2, axis=(1, 2)), axis=0)

        loss, grads = jax.value_and_grad(mse_loss)((poses, c, g), x_i, y_i)

        # Update the latent features
        c = c - config.optim.inner_lr_c * grads[1]

        # Update the poses if specific lr > 0
        if config.optim.inner_lr_p > 0:
            poses = poses - config.optim.inner_lr_p * grads[0]
        if config.optim.inner_lr_g > 0:
            g = g - config.optim.inner_lr_g * grads[2]

        # Return loss with resulting latents
        return mse_loss((poses, c, g), x_i, y_i), (poses, c, g)

    @jax.jit
    def outer_step(x_i, y_i, enf_params, enf_opt_state, key):
        # SPlit key
        key, new_key = jax.random.split(key)

        # Perform inner loop optimization
        (loss, _), grads = jax.value_and_grad(inner_loop, has_aux=True)(enf_params, x_i, y_i, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_optimizer.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return loss, enf_params, enf_opt_state, new_key

    # Training loop
    glob_step, lowest_loss = 0, jnp.inf
    for epoch in range(config.train.num_epochs):
        epoch_loss = []
        for i, batch in enumerate(train_dloader):
            # Unpack batch, flatten img
            img = batch[0]
            img = img + torch.randn_like(img)
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Perform outer loop optimization
            loss, enf_params, enf_opt_state, key = outer_step(
                x, y, enf_params, enf_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and log an image, perform inner loop
                _, (p_b, c, g) = inner_loop(enf_params, x, y, key)

                # Reconstruct image
                img_r = model.apply(enf_params, x, p_b, c, g)[0]

                # Min max norm
                min_val = jnp.min(img[0])
                max_val = jnp.max(img[0])

                img_r = (img_r - jnp.min(img_r)) / (jnp.max(img_r) - jnp.min(img_r))
                img = (img - min_val) / (max_val - min_val)

                print(img.min(), img.max(), img_r.min(), img_r.max())

                # Plot the original and reconstructed image
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(jnp.reshape(img[0], (img_shape)))
                plt.title("Original")
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(jnp.reshape(img_r, (img_shape)))
                plt.title("Reconstructed")
                plt.axis('off')

                # Plot the poses
                plt.subplot(133)
                plt.imshow(jnp.reshape(img_r, (img_shape)))
                plt.title("Poses")
                plt.axis('off')

                # Poses are [-1, 1], map to [0, img_shape]
                poses_m = (p_b + 1) / 2 * img_shape[0]
                plt.scatter(poses_m[0, :, 0], poses_m[0, :, 1], c='r')
                wandb.log({"ep_loss": sum(epoch_loss) / len(epoch_loss), "reconstructed": plt})
                plt.close('all')
                logging.info(f"epoch {epoch} -- loss: {sum(epoch_loss) / len(epoch_loss)}")

                # Save checkepcointointon if lowest loss
                if sum(epoch_loss) / len(epoch_loss) < lowest_loss:
                    lowest_loss = sum(epoch_loss) / len(epoch_loss)

                    checkpoint_manager.save(step=epoch, args=ocp.args.Composite(
                        state=ocp.args.StandardSave(enf_params),
                        config=ocp.args.JsonSave(config.to_dict())))

    run.finish()


if __name__ == "__main__":
    app.run(main)
