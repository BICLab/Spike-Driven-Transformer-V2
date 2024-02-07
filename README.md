# Spike-Driven Transformer ([NeurIPS2023](https://openreview.net/forum?id=9FmolyOHi5))

[Man Yao*](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Jiakui Hu*](https://github.com/jkhu29), [Tianxiang Hu](), [Yifan Xu](), [Zhaokun Zhou](), [Yonghong Tian](https://scholar.google.com/citations?user=fn6hJx0AAAAJ), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

BICLab, Institute of Automation, Chinese Academy of Sciences

---

:rocket:  :rocket:  :rocket: **News**:

- **Jan. 16, 2024**: Accepted as poster in ICLR2024.
- **Feb. 07, 2024**: Release the training and inference codes on IN1K.

## Abstract

Neuromorphic computing, which exploits Spiking Neural Networks (SNNs) on neuromorphic chips, is a promising energy-efficient alternative to traditional AI. CNN-based SNNs are the current mainstream of neuromorphic computing. By contrast, no neuromorphic chips are designed especially for Transformer-based SNNs, which have just emerged, and their performance is only on par with CNN-based SNNs, offering no distinct advantage. In this work, we propose a general Transformer-based SNN architecture, termed as ``Meta-SpikeFormer", whose goals are: (1) **Lower-power**, supports the spike-driven paradigm that there is only sparse addition in the network; (2) **Versatility**, handles various vision tasks; (3) **High-performance**, shows overwhelming performance advantages over CNN-based SNNs; (4) **Meta-architecture**, provides inspiration for future next-generation Transformer-based neuromorphic chip designs. Specifically, we extend the [Spike-driven Transformer](https://github.com/BICLab/Spike-Driven-Transformer) into a meta architecture, and explore the impact of structure, spike-driven self-attention, and skip connection on its performance. On ImageNet-1K, Meta-SpikeFormer achieves **80.0% top-1 accuracy** (55M), surpassing the current state-of-the-art (SOTA) SNN baselines (66M) by 3.7%. This is the first direct training SNN backbone that can simultaneously **supports classification, detection, and segmentation**, obtaining SOTA results in SNNs. Finally, we discuss the inspiration of the meta SNN architecture for neuromorphic chip design.

![V2](./img/300_spike_driven_transformer_v2_me.png)

## Requirements

```python3
pytorch >= 2.0.0
cupy
spikingjelly == 0.0.0.0.12
```

## Results on Imagenet-1K

The checkpoints are comming soon.

## Train & Test

The hyper-parameters are in `./conf/`.


Train:

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 128 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --epochs 200 \
  --model spikformer_8_512_CAFormer \
  --data_path /your/data/path \
  --output_dir outputs/T1 \
  --log_dir outputs/T1 \
  --model_mode ms \
  --dist_eval
```

Finetune:

> Please download caformer_b36_in21_ft1k.pth first.

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 24 \
  --blr 2e-5 \
  --warmup_epochs 5 \
  --epochs 50 \
  --model spikformer_8_512_CAFormer \
  --data_path /your/data/path \
  --output_dir outputs/T4 \
  --log_dir outputs/T4 \
  --model_mode ms \
  --dist_eval \
  --finetune /your/ckpt/path \
  --time_steps 4 \
  --kd \
  --teacher_model caformer_b36_in21ft1k \
  --distillation_type hard
```

Test:

```shell
python main_finetune.py --batch_size 128 --model spikformer_8_512_CAFormer --data_path /your/data/path --eval --resume /your/ckpt/path
```

Result and explainability:

![The Attention Map of Spike-Driven Transformer in ImageNet.](./imgs/Fig_3_attention_map.png)

## Data Prepare

- use `PyTorch` to load the CIFAR10 and CIFAR100 dataset.
- use `SpikingJelly` to prepare and load the Gesture and CIFAR10-DVS dataset.

Tree in `./data/`.

```shell
.
├── cifar-100-python
├── cifar-10-batches-py
├── cifar10-dvs
│   ├── download
│   ├── events_np
│   ├── extract
│   ├── frames_number_10_split_by_number
│   └── frames_number_16_split_by_number
├── cifar10-dvs-tet
│   ├── test
│   └── train
└── DVSGesturedataset
    ├── download
    ├── events_np
    │   ├── test
    │   └── train
    ├── extract
    │   └── DvsGesture
    ├── frames_number_10_split_by_number
    │   ├── download
    │   ├── test
    │   └── train
    └── frames_number_16_split_by_number
        ├── test
        └── train
```

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```
@inproceedings{
yao2024spikedriven,
title={Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips},
author={Man Yao and JiaKui Hu and Tianxiang Hu and Yifan Xu and Zhaokun Zhou and Yonghong Tian and Bo XU and Guoqi Li},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=1SIBN5Xyw7}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `manyao@ia.ac.cn` and `jkhu29@stu.pku.edu.cn`.