# ENF-Hackathon


## Resources
- [enf_standalone.ipynb](./enf_standalone.ipynb). This notebook can be used as a stand-alone implementation of ENF and contains code for fitting and classificiation. You could e.g. run this on Google Colab.
- [jax_tiny_intro.ipynb](./jax_tiny_intro.ipynb). This 
- [enf-min-jax](https://github.com/david-knigge/enf-min-jax). This repository contains example implementations of classification and diffusion on CIFAR10, as well as multi-modal segmentation on the OMBRIA satellite imagery dataset. It could be a good resource if you'd like to apply ENF to your own problem.
- [FIGURE](https://github.com/ebekkers/FIGURE). This repository contains a synthetic dataset for studying shape-based and motion-based representation learning, while controlling for texture bias and global transformations. It also contains baseline results for classification and generative modelling using CNNs, highlighting texture bias in conventional models. Could ENF be an outcome here?
- [Equivariant Neural Fields](https://arxiv.org/abs/2406.05753). This manuscript contains the formalization of ENF, as well as examples of use-cases. Might be useful as a reference.
- [PDE solving with Equivariant Neural Fields](https://arxiv.org/abs/2406.06660). This work uses ENF as a backbone for learning to forecast symmetric PDEs over general geometries. Might be useful as a reference.

## Running the example notebook
Assuming you have a GPU with CUDA 12 installed, you can install the dependencies with the following commands:

```bash
conda create -n enf-hackathon python=3.11
conda activate enf-hackathon
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections pillow h5py tqdm jupyter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Implemented datasets

### Double Pendulum

Spatiotemporal data 

#### To generate the dataset
```bash
python datasets/double_pendulum/generate_pendulum_dataset.py
```

#### To visualize the dataset
```bash
python datasets/double_pendulum/visualize_dataset.py
```