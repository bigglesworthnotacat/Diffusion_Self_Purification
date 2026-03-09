# Self-Purification Mitigates Backdoors in Multimodal Diffusion Language Models

**[Self-Purification Mitigates Backdoors in Multimodal Diffusion Language Models](https://arxiv.org/abs/2602.22246)**

**Guangnian Wan, Qi Li, Gongfan Fang, Xinyin Ma, Xinchao Wang**

*National University of Singapore*

## Introduction

We introduce **DiSP** (**Di**ffusion **S**elf-**P**urification), a backdoor defense for Multimodal Diffusion Language Models that masks vision tokens at inference to neutralize triggers, purifies the poisoned data with the compromised model itself, and fine-tunes it back to a clean state, without auxiliary models or clean reference data.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+

### Setup

```bash
git clone https://github.com/bigglesworthnotacat/DiSP.git
cd DiSP/LLaDA-V
bash init_env.sh
```

> Before running any script, update the path variables at the top of each `.sh` file.

## Training

Download the [LLaDA-V](https://huggingface.co/GSAI-ML/LLaDA-V) model, then finetune it on the poisoned dataset to inject the backdoor:

```bash
bash scripts/finetune.sh
```

## Evaluation

### Backdoor Evaluation (ASR)

Evaluate the Attack Success Rate (ASR) of a backdoored model on poisoned or clean images:

```bash
bash scripts/backdoor_eval.sh
```

### Clean Performance Evaluation (MMMU)

Evaluate the model's clean performance on the MMMU benchmark:

```bash
bash scripts/lmms_eval.sh
```

## Purification

### Data Purification

Re-caption the poisoned dataset using mask-based inference to neutralize backdoor effects:

```bash
bash scripts/self_purification.sh
```

### Recovery Fine-tuning on Purified Data

After self-purification, merge the adapter weights (`mm_projector.bin`) into the original model to produce a single merged checkpoint. Then fine-tune on the purified dataset to recover the clean model:

```bash
bash scripts/finetune_purified.sh
```

## Acknowledgement

Our code is based on [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) and we are grateful for their excellent work.
