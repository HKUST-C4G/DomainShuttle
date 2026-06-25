<div align="center">


<h1>DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation</h1>

[**Nan Chen†**](https://cn-makers.github.io/) ·**Yiyang Cai†** · **Rongchang Xie** · **Junwen Pan** · **Cheng Chen** · **Weinan Jia** · **Zhuowei Chen** ·

 **Wen Zhou‡** · **Zhenbang Sun** · **Wenhan Luo**<sup>&#9993;</sup>


<sup>†</sup>Equal Contribution.
<sup>‡</sup>Project Leader
<sup>&#9993;</sup>Corresponding Author


<a href='https://cn-makers.github.io/DomainShuttle/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2606.26058/'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/CNcreator0331/DomainShuttle_weight'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>

> **TL; DR:**  We propose **DomainShuttle**, an open-domain subject-driven text-to-video method that flexibly handles both in-domain fidelity and cross-domain editability by decoupling reference and video features, modeling domain attributes, and learning intrinsic subject representations.

<p align="center">
  <img src="asset/teaser.png">
</p>
## 🔥 Latest News


📖 *Jun 23, 2026:* We release the [technical report](https://arxiv.org/abs/2511.23475).

🔥 *Jun 23, 2026:* We release the unofficial **DomainShuttle** implementation [weights](#), [inference code](#), and [project page](https://hkust-c4g.github.io/DomainShuttle-homepage).


## 📑 Todo List
- [✅] Inference code
- [✅] checkpoint
- [✅] Technical report 

## Quick Start

### 🛠️Installation

#### 1. Create a conda environment and install pytorch
```
conda create -n DomainShuttle python=3.10
conda activate DomainShuttle 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

bash build_env_conda.sh
```


### 🧱Model Preparation

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.2-T2V-A14B  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/tree/main)       | Base model
| DomainShuttle      |      🤗 [Huggingface](https://huggingface.co/CNcreator0331/DomainShuttle_weight)              | Our 14B weights

Download models using huggingface-cli:
``` sh
mkdir models/Diffusion_Transformers
hf download CNcreator0331/DomainShuttle_weight --local-dir ./models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir ./checkpoints/Wan2.2-T2V-A14B
mv ./checkpoints/Wan2.2-T2V-A14B/google  ./models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B/
mv ./checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth ./models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B/
mv ./checkpoints/Wan2.2-T2V-A14B/configuration.json ./models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B/
mv ./checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth ./models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B/

```


The directory should be organized as follows.

```
models/Diffusion_Transformers/
└── Wan2.2-DomainShuttle-A14B/
    ├── google/
    ├── high_noise_model/
    ├── low_noise_model/
    ├── configuration.json
    └── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1_VAE.pth
    
``` 


### 🔑 Quick Inference
This unofficial inference script can run 480p/720p inference on GPU.


#### 14B inference
```sh
bash run_wan22_domainshuttle.sh
```


We provide some test cases in test_case folder. You can also try our model with your own data. You can change the reference image in the json file.







## 📚 Citation

If you find our work useful in your research, please consider citing.

## 📜 License
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 
