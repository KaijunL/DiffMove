# 5322
## Overview

This is the github repository for the submission 5322. We introduce a conditional diffusion-based trajectory recovery method. It first transforms locations in trajectories into the embedding space, in which the embedding denoising is performed, and then missing locations are recovered by an embedding decoder. The proposed model not only improves accuracy by introducing high-quality generative methods in the trajectory recovery, but also carefully models the transition, spatial, and temporal patterns in human mobility. 

![Figure_1](https://github.com/user-attachments/assets/7abdc991-0fa9-4591-a974-dc0bedbef130)



## Installation

Provide instructions on how to install the project. Include the commands to clone the repository and any necessary dependencies.

```bash
git clone <this_http_link>
cd 5322
pip install -r requirements.txt
```

## Usage

You can modify the config file and base.yml in diff_config folder, then run the python main.py

```bash
python main.py
```

## Dataset

Running with the open source dataset Foursqure(Tokyo) from https://sites.google.com/site/yangdingqi/home/foursquare-dataset/ (followed baselines' pre-processing steps to obtain the processed data). 


