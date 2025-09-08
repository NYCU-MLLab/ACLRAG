
# ACLRAG
Adversarial Contrastive Learning for Efficient RAG

## Environment Setup
We provide an `environments.yml` file for setting up the conda environment:

```bash
conda env create -f environments.yml
conda activate aclrag
````

> **Note**:
>
> * The provided `torch` version in `environments.yml` is for **NVIDIA CUDA**.
> * If you are using **AMD ROCm**, please install the ROCm-compatible `torch` manually. See [PyTorch ROCm](https://pytorch.org/get-started/locally/) for details.

---

## Datasets

* **Pretraining Dataset**: [The Pile](https://pile.eleuther.ai/)
* **Finetuning Dataset**: [HotpotQA](https://hotpotqa.github.io/)

---

## Training

### 1. Reconstruction Pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py
```

### 2. Query-Aware Finetuning (Contrastive + Adversarial Learning)

```bash
CUDA_VISIBLE_DEVICES=0 python contrastive_learning.py
```

---

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python fine_tuned_inference_ACLRAG.py
```

