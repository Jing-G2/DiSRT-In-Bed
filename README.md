# DiSRT-In-Bed: Diffusion-Based Sim-to-Real Transfer Framework for In-Bed Human Mesh Recovery


<a href="http://arxiv.org/abs/2504.03006"><img src="https://img.shields.io/badge/CVPR 2025 Paper-arXiv-b31b1b.svg" height=22></a>
<a href="https://jing-g2.github.io/DiSRT-In-Bed/"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https://jing-g2.github.io/DiSRT-In-Bed/" height=22></a>

## Installation

### 1. Set up conda environment
```
conda create -n bed python=3.10
conda activate bed
pip install -r requirements.txt
```

### 2. Install external dependencies

Follow instructions from [shapy](https://github.com/muelea/shapy/blob/master/documentation/INSTALL.md#code) for shape metrics calculation.

```
mkdir external
cd external
git clone https://github.com/muelea/shapy.git
```

- Install python dependencies

```
pip install chumpy imageio jpeg4py joblib kornia loguru matplotlib numpy open3d Pillow PyOpenGL pyrender scikit-image scikit-learn scipy smplx tqdm trimesh yacs fvcore nflows PyYAML
pip install omegaconf
pip install pytorch-lightning
```

- Install local dependencies

```
cd shapy/attributes
pip install .

cd ../mesh-mesh-intersection
export CUDA_SAMPLES_INC=$(pwd)/include
pip install -r requirements.txt
pip install .
```

## Data Setup

1. Follow the instructions in [BodyMAP](https://github.com/RCHI-Lab/BodyMAP) to download SLP dataset, BodyPressureSD dataset, and SMPL human models.

## Run

Our proposed DiSRT-In-Bed framework consists of two stages: training and finetuning.

- Training: run the `train.sh` script to train the diffusion model on synthetic data.
    - delete the `--viz_type` line in the `train.sh` script if you do not want to visualize the training process.
- Finetuning: run the `finetune.sh` script to finetune the model on real data.
- Evaluation: change the `mode` in the finetune script to `test` to evaluate the model on the test set.
    - if you want to evaluate the model on hospital dataset, change the `exp_type` to `hospital` in the `finetune.sh` script.

## Acknowledgements
Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.

- [BodyMAP: Jointly Predicting Human Body Mesh and Pressure Maps from Depth and Pressure Images](https://github.com/RCHI-Lab/BodyMAP)
- [BodyPressure - Inferring Body Pose and Contact Pressure from a Depth Image](https://github.com/Healthcare-Robotics/BodyPressure)
- [guided-diffusion (Diffusion Models Beat GANs on Image Synthesis)](https://github.com/openai/guided-diffusion)
- [Closely Interactive Human Reconstruction with Proxemics and Physics-Guided Adaption (CloseInt)](https://github.com/boycehbz/HumanInteraction)
