# ENF-Hackathon

## 🚀 Introduction
Welcome to the **ENF-Hackathon**! This repository contains an example implementation of **Equivariant Neural Fields (ENF)** for downstream tasks. During this hackathon, you'll explore and tackle one of the challenges discussed this morning, such as improving position updates or incorporating hierarchy into latent point clouds.

🔍 This codebase includes a **notebook** demonstrating how to fit an equivariant neural field and train a classifier on latent point clouds.

### 📊 Performance Evaluation
We will compare different solutions in **two ways**:
1. **Downstream Classifier** – Evaluating how meaningful the latent point clouds are.
2. **Generative Model** – Understanding how well latent representations capture information.

💡 **Note:** Directly comparing ENFs to pixel-based models (e.g., CNNs) is not entirely fair since **ENFs are domain-agnostic**, not designed specifically for the image domain. Focus on improving **ENF-based performance** rather than competing with CNN methods.

We provide three datasets to experiment with:
- **FIGURE** (synthetic shape-based & motion-based dataset)
- **CIFAR-10** (natural image dataset)
- **Double Pendulum** (spatiotemporal dataset)

---

## 🧩 Problems & Research Questions
Here are key challenges and ideas for improvement:

### ⚠️ Current Challenges
| Problem | Potential Solution |
|---------|---------------------|
| **Underpowered decoder with many/large latents** | Multi-layer ENF? Self-attention between latents? |
| **MAML struggles with latent pose inference (local minima?)** | Replace SGD with an encoder? |
| **Limited performance of downstream models** | Additional structure in ENF latent space? A VAE variant? |
| **Current latents are forcibly local, some tasks need global features** | Introduce hierarchy? Global/local latent separation? |
| **Unclear how to represent spatiotemporal data with ENF** | Predictor-corrector scheme (e.g., SAVI++, MooG)? |

### ❓ Research Question
- **How does the choice of RFF embedding (periodic) affect reconstruction and downstream performance?**

---

## 📚 Resources
Here are useful references to help you get started:

📌 **Code & Notebooks**
- [🔗 `enf_standalone.ipynb`](./enf_standalone.ipynb) – A standalone ENF implementation for fitting & classification. Can be run on Google Colab.
- [🔗 `jax_tiny_intro.ipynb`](./jax_tiny_intro.ipynb) – A short introduction to JAX, the library used for automatic differentiation in ENF.

📌 **Relevant Repositories**
- [🛠️ `enf-min-jax`](https://github.com/david-knigge/enf-min-jax) – Example implementations of classification, diffusion (CIFAR10), and multi-modal segmentation (OMBRIA dataset).
- [📊 `FIGURE`](https://github.com/ebekkers/FIGURE) – A dataset for studying shape/motion-based representation learning, highlighting texture bias in CNNs. Can ENF mitigate this?

📌 **Research Papers**
- [📜 Equivariant Neural Fields](https://arxiv.org/abs/2406.05753) – The formalization of ENF, including various use cases.
- [📜 PDE Solving with ENF](https://arxiv.org/abs/2406.06660) – Application of ENF for forecasting symmetric PDEs over general geometries.

---

## 🛠️ Running the Example Notebook
To set up your environment, follow these steps (**Requires CUDA 12** for GPU acceleration):

```bash
conda create -n enf-hackathon python=3.11
conda activate enf-hackathon
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections pillow h5py tqdm jupyter
tpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🌌 Extra Dynamics Dataset: Double Pendulum
We provide a **spatiotemporal dataset** of a **double pendulum** for experimentation.

### 📥 Dataset Generation
Run the following command to generate the dataset:
```bash
python datasets/double_pendulum/generate_pendulum_dataset.py
```

### 📊 Dataset Visualization
To visualize the dataset, use:
```bash
python datasets/double_pendulum/visualize_dataset.py
```

---

🎯 **Let’s innovate together and push the boundaries of ENF! 🚀**

