# Unified SAM Optimization Frameworks: A PyTorch Library for Sharpness-Aware Minimization

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository provides a **unified PyTorch implementation** of **Sharpness-Aware Minimization (SAM)** and its various derivatives. 
It offers a consistent training and evaluation interface, aiming to facilitate easy **comparison, reproducibility, and extension** of SAM-style optimizers across standard vision architectures.
Therefore, the goal of this repository is to make it easy to **compare, reproduce, and extend** SAM-style optimizers across common vision architectures and datasets.

---

## Requirements:

> **Note**: This repository works with most **Python versions** and **PyTorch >= 2.0**.  
> The commands below provide a tested setup (Python 3.9 + PyTorch 2.3.1, CUDA 11.8).

### 1) Create a Conda environment
```bash
conda create -n SAM python=3.9 -y
conda activate SAM
```

### 2) Install PyTorch
```bash
# Conda Version
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip Version
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## Features

### Supported Optimizers
| Optimizer | Short Description | Conference | Paper |
| :--- | :--- | :---: | :---: |
| **SAM** | Sharpness-Aware Minimization | **`ICLR'21`** | 📄 [ArXiv](https://arxiv.org/abs/2010.01412) |
| **ASAM** | Adaptive | **`ICML'21`** | 📄 [ArXiv](https://arxiv.org/abs/2102.11600) |
| **ESAM** | Efficient | **`ICLR'22`** | 📄 [ArXiv](https://arxiv.org/abs/2110.03141) |
| **GSAM** | Surrogate Gap (*In Progress*) | **`ICLR'22`** | 📄 [ArXiv](https://arxiv.org/abs/2203.08065) |
| **FisherSAM** | Fisher Information | **`ICML'22`** | 📄 [ArXiv](https://arxiv.org/abs/2206.04920) |
| **BayesianSAM** | Bayesian Perspective | **`ICLR'23`** | 📄 [ArXiv](https://arxiv.org/abs/2210.01620) |
| **F-SAM** | Friendly | **`CVPR'24`** | 📄 [ArXiv](https://arxiv.org/abs/2403.12350) |
| **LookSAM** | Lookahead | **`ICML'24`** | 🌐 [OpenReview](https://openreview.net/pdf?id=J9YKDvqr65) |


### Supported Architectures
- **ResNet**: 18, 34, 50, 101, 152
- **WideResNet**: 28, 34
- **PyramidNet**

### Supported Datasets
- **CIFAR-10**
- **CIFAR-100**
- **TinyImageNet**

---

## Usage & Examples

Below are example scripts to reproduce results for various SAM variants.

### SAM

```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --rho 0.05 --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --rho 0.05 --seed 1234

# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --adaptive --rho 0.05 --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --adaptive --rho 0.05 --seed 1234
```

### ASAM
```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --adaptive --rho 0.05 --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --adaptive --rho 0.05 --seed 1234
```

### ESAM
```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type ESAM --rho 0.05 --beta 0.5 --gamma 0.5 \
  --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type ESAM --rho 0.05 --beta 0.5 --gamma 0.5 \
  --seed 1234
```

### GSAM (TODO)
> GSAM is currently under refactoring and not fully supported in this repository.

### FisherSAM
```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type FisherSAM --rho 0.05 --eta 0.2 \
  --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type FisherSAM --rho 0.05 --eta 0.2 \
  --seed 1234
```

### BayesianSAM
```bash
# CIFAR-10 (ResNet-18)
python main.py --dataset cifar10 --arch_type resnet18 --optimizer sgd \
  --epochs 180 --batch_size 200 --lr 0.5 --weight_decay 0 \
  --sam_type BayesianSAM --msharpness 8 --gamma 0.1 --rho 0.01 --delta 10 \
  --seed 1234

# CIFAR-100 (ResNet-34)
python main.py --dataset cifar100 --arch_type resnet34 --optimizer sgd \
  --epochs 180 --batch_size 200 --lr 0.5 --weight_decay 0 \
  --sam_type BayesianSAM --msharpness 8 --gamma 0.1 --rho 0.01 --delta 10 \
  --seed 1234
```

### Friendly-SAM
```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 \
  --sam_type FriendlySAM --rho 0.05 --sigma 1.0 --lmbda 0.9 \
  --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 \
  --sam_type FriendlySAM --rho 0.05 --sigma 1.0 --lmbda 0.9 \
  --seed 1234
```

### Look(-Layer)SAM
```bash
# CIFAR-10 (WideResNet-28)
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 \
  --sam_type LookSAM --rho 0.05 --k 5 --alpha 1.0 \
  --seed 1234

# CIFAR-100 (PyramidNet)
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 \
  --sam_type LookSAM --rho 0.05 --k 5 --alpha 1.0 \
  --seed 1234
```

---

## Future Works / TODO (In Progress)

### Reproduction & Benchmarking
- [ ] **Reproduce Results Table:** Reproduce and tabulate results using optimal hyperparameters from original papers to ensure fair comparison.
- [ ] **Hyperparameter Tuning:** Find and verify optimal hyperparameters (e.g., `rho`, learning rate) for each variant.

### Development
- [ ] **Modify GSAM Code:**
  - Refactor GSAM to align with the common optimizer interface used in this repository.
  - Verify correctness (loss/accuracy parity with reference) and add a minimal reproducible run config.
- [ ] **Add ViT-based Models:**
  - Implement ViT-family backbones (e.g., ViT, DeiT) under the same model API.
  - Ensure compatibility with SAM-style optimizers (specifically addressing BN-free / LayerNorm behavior).
- [ ] **Add ImageNet Dataset:**
  - Implement ImageNet dataloaders and standard augmentation pipelines.
  - Provide standard training/evaluation configs.
- [ ] **Apply DDP (DistributedDataParallel):**
  - Add multi-GPU training support with PyTorch DDP.
  - Ensure deterministic logging and checkpointing across processes.
  - Validate SAM/variant behavior under DDP (specifically handling gradient synchronization and second-step updates).

### Extensions
- [ ] **Additional Approaches:** Feel free to contact me or open an issue to suggest/add additional SAM approaches.

---

## Code References (Github) :

| Optimizer | Implementation / Code | Status |
| :--- | :---: | :---: |
| **SAM** | [JAX](https://github.com/google-research/sam) / [PyTorch](https://github.com/davda54/sam) | ✅ |
| **ASAM** | [PyTorch](https://github.com/davda54/sam) | ✅ |
| **ESAM** | [PyTorch](https://github.com/dydjw9/Efficient_SAM/) | ✅ |
| **GSAM** | [JAX](https://github.com/google-research/big_vision/tree/main/big_vision/trainers/proj/gsam)  / [PyTorch](https://github.com/juntang-zhuang/GSAM) | 🚧 |
| **FisherSAM** |`Reproduced` | ✅ |
| **BayesianSAM** | [JAX](https://github.com/team-approx-bayes/bayesian-sam) | ✅ |
| **F-SAM** | [PyTorch](https://github.com/nblt/F-SAM) | ✅ |
| **LookSAM** | [PyTorch](https://github.com/rollovd/LookSAM/) | ✅ |

## Acknowledgements:

This repository is built upon and inspired by the following excellent open-source projects. 
I sincerely thank the original authors and contributors for their valuable work.