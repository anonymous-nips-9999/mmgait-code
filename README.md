# Anonymous Codebase for NeurIPS 2025 Submission

This repository provides an anonymized implementation of a **single-modality gait recognition framework**, built upon the [OpenGait](https://github.com/ShiqiYu/OpenGait) codebase.

At this stage, only the training and evaluation pipeline for single-modality gait recognition is released. The full codebase, including multi-modality extensions, will be released upon paper acceptance.

## 🚀 Quick Start

To reproduce single-modality recognition experiments, please run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 opengait/main.py \
  --cfgs ./configs/mmgait/DeepGaitV2_mmgait.yaml \
  --phase train
