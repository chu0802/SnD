<div align="center">

# Select and Distill

This is an official implementation of our work, Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models, accepted to ECCV'24.
  
[![PWC](https://img.shields.io/badge/arXiv-2403.09296-b31b1b)](https://arxiv.org/abs/2403.09296)
[![PWC](https://img.shields.io/badge/ECCV%202024-PDF-FACE27)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03759.pdf)
[![PWC](https://img.shields.io/badge/ECCV%202024-Supp-7DCBFF)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03759-supp.pdf)
[![PWC](https://img.shields.io/badge/ECCV%202024-Bibtex-CB8CEA)](#citation)

https://chuyu.org/research/snd


</div>


## Table of Contents
- [Announcement](#announcement)
- [Installation](#install)
- [Data Preparation](#data)
- [Running the model](#run)
- [Citation](#citation)

<a name="announcement"></a>
## Annoucement

**[2025/01/19]** The instruction page is ready! We plan to release our original checkpoints soon.

**[2024/12/31]** Our full codebase has been released! Introduction and installation method (include packages) would be updated soon.

<a name="install"></a>
## Installation

Create a new Conda environment with Python 3.10.14:

```
conda create -n snd python==3.10.14
```

Activate the environment and install PyTorch with the specified version and CUDA support:

```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install additional dependencies using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

<a name="data"></a>
## Dataset Preparation

To reproduce our experiments, download the following datasets from the guidance provided [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

- FGVCAircraft
- DTD
- EuroSAT
- Flowers102
- Food101
- OxfordPets
- StanfordCars
- UCF101
- ImageNet


Organize each dataset in the following directory structure:

```
<DATASET_NAME>/
    ├── images/
    │   ├── image data / folders
    ├── <DATASET_NAME>_annotations.json
```

The `<DATASET_NAME>_annotations.json` file contains the training, validation, and test splits, along with class names. The files we used for all datasets are provided [here](https://drive.google.com/drive/folders/1q3sW405OXYNO1E57Hu-NaEvF_FopRhQ8?usp=sharing). Download these files and place them in the appropriate paths as described above.

<a name="run"></a>
## Running with the Scripts

We provide several scripts to help you easily reproduce our experiments. Note that our experiments were conducted using 4x V100 GPUs in distributed parallel mode. Note that we have not tested our method outside of distributed mode, while the distributed mode can run on a single GPU.

### Train and Eval

The following script allows training on **a single dataset** (e.g., fgvc-aircraft) and evaluating on **all datasets** using 4 GPUs.

Run the command below to execute the script:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_4_gpus.yaml --dataset fgvc-aircraft --distributed --nproc_per_node 4
```

#### Using a Single GPU

If you are using only one GPU, modify the command as follows:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_1_gpu.yaml --dataset fgvc-aircraft --distributed --nproc_per_node 1
```

#### Continual Training

To load a model trained on a specific dataset and **continue training** on another dataset, include the `--pretrained_dataset` argument:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_4_gpus.yaml --pretrained_dataset fgvc-aircraft --dataset dtd --distributed --nproc_per_node 4
```

#### Note

- Our code has only been verified with 1 or 4 GPUs.
- Using more than 4 GPUs is not recommended, as we observed that the performance drops a bit.
- When training with 1–4 GPUs, ensure that the batch size for training and reference data is correctly adjusted to match the number of GPUs.

### Continual Training on the whole training sequence

We also provide a script to continually train and evaluate across an entire sequence of datasets (i.e., reproduce our Multi-Domain Task Incremental Learning setting):

```sh
python -m scripts.continually_train --config_path configs/snd_config_4_gpus.yaml --order 0 --distributed --nproc_per_node 4
```

#### Note

- The `--order` argument specifies an offset to shift the pre-defined dataset sequence.
- For detailed task orders of each training sequence, refer to the supplementary materials.

<a name="citation"></a>
## Citation

If you find our work useful, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{yu2025select,
  title={Select and distill: Selective dual-teacher knowledge transfer for continual learning on vision-language models},
  author={Yu, Yu-Chu and Huang, Chi-Pin and Chen, Jr-Jen and Chang, Kai-Po and Lai, Yung-Hsuan and Yang, Fu-En and Wang, Yu-Chiang Frank},
  booktitle={European Conference on Computer Vision},
  pages={219--236},
  year={2025},
  organization={Springer}
}
```
