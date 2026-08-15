# DiffMove for ICML 2025 Paper: Generative Human Trajectory Recovery via Embedding-Space Conditional Diffusion 

Official implementation of our **ICML 2025** paper:

> **Generative Human Trajectory Recovery via Embedding-Space Conditional Diffusion**  
> Kaijun Liu, Sijie Ruan, Liang Zhang, Cheng Long, Shuliang Wang, Liang Yu  
> *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*

<p align="center">
  <a href="https://proceedings.mlr.press/v267/liu25bg.html">📄 Paper</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25bg/liu25bg.pdf">📑 PDF</a>
</p>

## Abstract

Recovering human trajectories from incomplete or missing data is crucial for many mobility-based urban applications, e.g., urban planning, transportation, and location-based services. Existing methods mainly rely on recurrent neural networks or attention mechanisms. Though promising, they encounter limitations in capturing complex spatial-temporal dependencies in low-sampling trajectories. Recently, diffusion models show potential in content generation. However, most of proposed methods are used to generate contents in continuous numerical representations, which cannot be directly adapted to the human location trajectory recovery. In this paper, we introduce a conditional diffusion-based trajectory recovery method, namely, DiffMove. It first transforms locations in trajectories into the embedding space, in which the embedding denoising is performed, and then missing locations are recovered by an embedding decoder. DiffMove not only improves accuracy by introducing high-quality generative methods in the trajectory recovery, but also carefully models the transition, periodicity, and temporal patterns in human mobility. Extensive experiments based on two representative real-world mobility datasets are conducted, and the results show significant improvements (an average of 11% in recall) over the best baselines.

## Citation

If you find this repository or our work useful in your research, please consider citing our paper:

```bibtex
@InProceedings{pmlr-v267-liu25bg,
  title     = {Generative Human Trajectory Recovery via Embedding-Space Conditional Diffusion},
  author    = {Liu, Kaijun and Ruan, Sijie and Zhang, Liang and Long, Cheng and Wang, Shuliang and Yu, Liang},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {39366--39380},
  year      = {2025},
  editor    = {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--19 Jul},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25bg/liu25bg.pdf},
  url       = {https://proceedings.mlr.press/v267/liu25bg.html}
}
```

## Overview

This is the github repository for DiffMove. We introduce a conditional diffusion-based trajectory recovery method. It first transforms locations in trajectories into the embedding space, in which the embedding denoising is performed, and then missing locations are recovered by an embedding decoder. The proposed model not only improves accuracy by introducing high-quality generative methods in the trajectory recovery, but also carefully models the transition, spatial, and temporal patterns in human mobility. 

![Alt text](Figure_1.png)



## Installation

Provide instructions on how to install the project. Include the commands to clone the repository and any necessary dependencies.

```bash
git clone <this_http_link>
cd DiffMove
pip install -r requirements.txt
```

## Usage

You can modify the config file and base.yml in diff_config folder, then run the python main.py

```bash
python main.py
```

## Dataset

Running with the open source dataset Foursqure(Tokyo) from https://sites.google.com/site/yangdingqi/home/foursquare-dataset/ (followed baselines' pre-processing steps to obtain the processed data). 


