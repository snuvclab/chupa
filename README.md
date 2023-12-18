# Chupa (ICCV 2023, Oral)

## [Project Page](https://snuvclab.github.io/chupa/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2305.11870.pdf) 

![teaser.png](./assets/teaser.png)

This is the official code for the ICCV 2023 paper "Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models", a 3D generation pipeline specialized on generating realistic human digital avatars.

## News
- [2023/08/20] Update text-based normal map generation checkpoint (1000 epochs).
- [2023/12/18] Release training script.

## Installation
Setup the environment using conda.
```
conda env create -f environment.yaml
conda activate chupa
```
Please check the installed pytorch has correct cuda version depending on your gpu.

#### Install pyremesh
```
python -m pip install --no-index --find-links ./src/normal_nds/ext/pyremesh pyremesh
```

### Get SMPL related data
Following the instructions from [ECON](https://github.com/YuliangXiu/ECON/blob/master/docs/installation-ubuntu.md), you should register [ICON website](https://icon.is.tue.mpg.de/).
Then, please run the script originally from [ECON](https://github.com/YuliangXiu/ECON/tree/master).
```
bash scripts/fetch_data.sh
```


### Install frankmocap (Optional, for running Gradio)
```
mkdir third_party
cd third_party
git clone https://github.com/facebookresearch/frankmocap.git
cd frankmocap
sh scripts/install_frankmocap.sh
mv extra_data ../../  # for relative path problem

# create setup.py and install frankmocap
echo "from setuptools import find_packages, setup

setup(
    name='frankmocap',
    packages=find_packages('.'),
    package_dir={'': '.'},
)" > ./setup.py
pip install -e .
```

## Training (23/12/18 updated)
We provide a sample training script using THuman2.0 dataset since we cannot share RenderPeople dataset due to license problem. You can follow the same process for the different human 3D scan dataset.
### Render normal maps
Please checkout [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) for getting 3D scans and SMPL-X parameters of THuman 2.0 dataset. Then please organize the data folder as following.
```
./data/
├── thuman/
│   └── scans/
│       └── 0000/
│           └── 0000.obj
│           └── material0.mtl
│           └── material0.jpeg
│       └── 0001/
│       └── ...
│   └── smplx/
│       └── 0000.pkl
│       └── 0001.pkl
│       └── ...
```

Then, please running following script to render the normal maps of 3D scans and fitted SMPL-X meshes.
```
bash scripts/render_dataset.sh thuman train
bash scripts/render_dataset.sh thuman test
```
You will have the following tree.
```
./data/
├── thuman/
│   └── render/
│       └── train/
│           └── 0000
│               └── normal_F
│                   └── 000.png
│                   └── 010.png
│                   └── ...
│                   └── 350.png
│               └── normal_face_F
│                   └── ...
│               └── T_normal_F
│                   └── ...
│               └── T_normal_face_F
│                   └── ...
│       └── test/
│           └── 0473
│               └── ...
```
`T_*` indicates rendering of SMPL-X. `normal_F` and `T_normal_F` will be used for training body normal map diffusion model, and `normal_face_F` and `T_normal_face_F` will be used for training face normal map diffusion model.
### Train a diffusion model with normal maps.
To train diffusion model with normal maps, please run the training code as below.
```
python src/ldm/main.py --base src/ldm/configs/thuman.yaml -t --device "${DEVICE IDs}"  # body diffusion
python src/ldm/main.py --base src/ldm/configs/thuman_face.yaml -t --device "${DEVICE IDs}"  # face diffusion
```
If you don't want to refine the mesh with zooming the face region, you may train only the body diffusion model. In this case, you should add `chupa.use_closeup=false` at the end when running the inference code. 

The training result will be saved under `src/ldm/logs`. Please move the desired checkpoint and config file (`configs/*-project.yaml`) to `checkpoints/normal_ldm` folder. Please refer to the pretrained model instruction below for the directory structure.

### Pretrained Model
At the moment, you can get the pretrained checkpoints by running the commands below. (23/12/18 the text model checkpoint was updated.)
The checkpoints include autoencoder checkpoints from [Latent Diffusion Model](https://github.com/CompVis/latent-diffusion).
```
gdown https://drive.google.com/uc?id=1N1n9MWdnrNANFZvwaCRx7oyX8LPnIteu  # Models for dual normal map generation
unzip checkpoints.zip && rm checkpoints.zip
mkdir checkpoints/autoencoder/vq-f4-c3 && cd checkpoints/autoencoder/vq-f4-c3
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip  # Autoencoder checkpoint from latent diffusion
unzip vq-f4.zip && rm vq-f4.zip 
```

## Inference
### Run Chupa on Gradio
You can upload an image of a person, then frankmocap will predict the SMPL-X parameter of it.  
Based on the predicted parameter, Chupa pipeline will generate a 3D human avatar (takes a few minutes).
(Note: Please ensure that the uploaded image contains one person.)
```
python scripts/gradio.py configs/gradio.yaml
```
![gradio.png](./assets/gradio.png)

### Run Chupa on Terminal
#### Getting Data
You can get the test SMPL-X parameters for the test split of renderpeople dataset.
```
mkdir data && cd data
gdown https://drive.google.com/uc?id=1pWsSEUIoHF_4Zjn_2OczC9FKdLjChWDY
unzip renderpeople.zip && rm renderpeople.zip
```
We followed the data split of [PIFuHD](https://github.com/facebookresearch/pifuhd/tree/main/data), and we release the SMPL-X parameters for test split at the moment. Please unzip it and follow the tree below.
```
./data/
├── renderpeople/
│   └── smplx/
│       └── rp_ben_posed_001.pkl
│       └── ...
```
You may checkout [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) for getting SMPL-X parameters for THuman2.0 dataset. We used `0473~0525` as test split.
#### Random generation
This will generate random 3D avatars based on SMPL-X parameters in `data/renderpeople/smplx`.
```
python scripts/chupa.py configs/random.yaml
```
Specify a subject in dataset
```
python scripts/chupa.py configs/random.yaml dataset dataset.subject=rp_ben_posed_001
``` 

#### Text-guided generation
This will generate 3D avatars based on the given prompt and SMPL-X parameters in `data/renderpeople/smplx`. 
```
python chupa.py --config configs/text.yaml chupa.prompt=${Your Prompt}
```

## Citation

If you use this code for your research, please cite our paper:


```
@InProceedings{kim2023chupa,
    author    = {Kim, Byungjun and Kwon, Patrick and Lee, Kwangho and Lee, Myunggi and Han, Sookwan and Kim, Daesik and Joo, Hanbyul},
    title     = {Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {15965-15976}
}
```

## Thanks to
- https://github.com/CompVis/latent-diffusion
- https://github.com/fraunhoferhhi/neural-deferred-shading
- https://github.com/YuliangXiu/ECON.git
- https://github.com/YuliangXiu/ICON.git
- https://github.com/facebookresearch/frankmocap
- https://github.com/crowsonkb/k-diffusion.git