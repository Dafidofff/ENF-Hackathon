# ENF-Hackathon

## Installation

```bash
conda create -n enf python=3.11
conda activate enf
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections pillow h5py tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Running the code

```bash
python fit_enf.py --config=configs/enf_config.py
```



### Double Pendulum

#### To generate the dataset
```bash
python datasets/double_pendulum/generate_pendulum_dataset.py
```

#### To visualize the dataset
```bash
python datasets/double_pendulum/visualize_dataset.py
```