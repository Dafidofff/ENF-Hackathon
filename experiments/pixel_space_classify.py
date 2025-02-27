from absl import app
from tqdm import tqdm

import wandb
import ml_collections
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False

    # Define the CNN model parameters
    config.cnn = ml_collections.ConfigDict()
    config.cnn.num_classes = 10
    config.cnn.features = [32, 64, 128]  # Feature sizes for each convolutional layer

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "./data"
    config.dataset.name = "cifar10"
    config.dataset.num_signals_train = 1000
    config.dataset.num_signals_test = 1000
    config.dataset.batch_size = 32
    config.dataset.num_workers = 8

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.lr = 1e-3
    config.train.num_epochs = 50
    config.train.log_interval = 5

    config.task = "classification"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(self.features, kernel_size=(1, 1), padding="SAME")(residual)

        x = nn.relu(x + residual)
        return x


class ResidualCNN(nn.Module):
    num_classes: int
    features: list

    @nn.compact
    def __call__(self, x):
        for features in self.features:
            x = ResidualBlock(features=features)(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


def main(_):
    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project="enf-hack", job_type="train", config=config.to_dict(), mode="online" if not config.debug else "dryrun")

    ##############################
    # Initializing the model
    ##############################

    # Load dataset, get sample image
    from datasets import get_dataloader # moved import here to avoid circular import issues.
    train_dloader, test_dloader = get_dataloader(config.dataset)
    sample = next(iter(train_dloader))
    img_shape = sample[0].shape[1:]

    # Random key
    key = jax.random.PRNGKey(config.seed)

    # Define the model
    model = ResidualCNN(num_classes=config.cnn.num_classes, features=config.cnn.features)

    # Initialize the model
    key, subkey = jax.random.split(key)
    dummy_img = jnp.ones((config.dataset.batch_size, *img_shape), dtype=jnp.float32)
    model_params = model.init(subkey, dummy_img)["params"]

    # Define optimizer
    optimizer = optax.adam(learning_rate=config.train.lr)
    opt_state = optimizer.init(model_params)

    @jax.jit
    def train_step(img, labels, model_params, opt_state, key):
        def loss_fn(model_params, img, labels):
            logits = model.apply({"params": model_params}, img)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(model_params, img, labels)

        updates, opt_state = optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)

        return loss, model_params, opt_state, key

    @jax.jit
    def eval_step(img, labels, model_params):
        logits = model.apply({"params": model_params}, img)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        return accuracy

    # Training loop
    glob_step = 0
    epoch_loss = []
    for epoch in tqdm(range(config.train.num_epochs), desc="Training epochs"):
        epoch_loss = []
        for i, batch in enumerate(train_dloader):
            img, labels = batch[0], batch[1]

            loss, model_params, opt_state, key = train_step(
                img, labels, model_params, opt_state, key
            )

            epoch_loss.append(loss)
            glob_step += 1

            # Logging
            wandb.log({"classification-loss": loss}, step=glob_step)

        if epoch % config.train.log_interval == 0:
            # Evaluate on test set
            accuracies = []
            for batch in test_dloader:
                test_img, test_labels = batch[0], batch[1]
                accuracy = eval_step(test_img, test_labels, model_params)
                accuracies.append(accuracy)
            test_accuracy = jnp.mean(jnp.array(accuracies))

            # Log the epoch loss and test accuracy
            wandb.log({"epoch-classification-loss": sum(epoch_loss) / len(epoch_loss)}, step=glob_step, commit=False)
            wandb.log({"test-accuracy": test_accuracy}, step=glob_step)
            epoch_loss = []

    run.finish()


if __name__ == "__main__":
    app.run(main)