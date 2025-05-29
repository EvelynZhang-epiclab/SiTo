# <p align="center"> SiTo: Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning (AAAI-2025)</p>

<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_logo.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:50%">
  <br>
</p>


## 🔥 News
- `2024/12/10`🤗🤗 Our [paper](https://www.researchgate.net/publication/387204421_Training-Free_and_Hardware-Friendly_Acceleration_for_Diffusion_Models_via_Similarity-based_Token_Pruning) is accepted by AAAI-2025
- `2025/1/18` 💥💥 We release the [code](https://github.com/EvelynZhang-epiclab/SiTo) for our work about accelerating diffusion models for FREE. 🎉 **The zero-shot evaluation shows SiTo leads to 1.90x and 1.75x acceleration on COCO30K and ImageNet with 1.33 and 1.15 FID reduction at the same time. Besides, SiTo has no training requirements and does not require any calibration data, making it plug-and-play in real-world applications.**
- `2025/1/21` 💿💿 We publish the required image for running our code. Click [here](https://www.codewithgpu.com/i/EvelynZhang-epiclab/SiTo/SiTo-SD) to use it directly.

## 🌸 Abstract
<details>

<summary> CLICK for full abstract </summary>

> The excellent performance of diffusion models in image generation is always accompanied by overlarge computation costs, which have prevented the application of diffusion models in edge devices and interactive applications. Previous works mainly focus on using fewer sampling steps and compressing the denoising network of diffusion models, while this paper proposes to accelerate diffusion models by introducing **SiTo, a similarity-based token pruning method** that adaptive prunes the redundant tokens in the input data. SiTo is designed to maximize the similarity between model prediction with and without token pruning by using cheap and hardware-friendly operations, leading to significant acceleration ratios without performance drop, and even sometimes improvements in the generation quality. For instance, **the zero-shot evaluation shows SiTo leads to 1.90x and 1.75x acceleration on COCO30K and ImageNet with 1.33 and 1.15 FID reduction** at the same time. Besides, SiTo has **no training requirements and does not require any calibration data**, making it plug-and-play in real-world applications.

</details>


## 🚀Overview

SiTo has a three-stage pipeline. 
- SiTo carefully selects a set of **base tokens** which are utilized as the base to select and recover the pruned tokens.
- SiTo selects the tokens that have the highest similarity to the base tokens as the **pruned tokens**.
- SiTo feeds the unpruned tokens to the neural layers and **recovers the pruned tokens** by directly copying their most similar base tokens.

<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_overview.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>The pipeline of SiTo on the example of self-attention. (a) Base Token Selection: We compute the Cosine Similarity between all the tokens. For each token, we sum its similarity to all the tokens as the SimScore. Then, Gaussian Noise is added to the SimScore introduces randomness, preventing identical base and pruned token choices across timesteps. Finally, the token that has the highest Noise SimScore in an image patch is selected as a base token. (b) Pruned Token Selection: The tokens with the highest similarity to the base tokens are selected as pruned tokens. (c) Pruned Token Recovery: The unpruned tokens are fed to the neural layers. Then, the pruned tokens are recovered by copying from their most similar base tokens.</em>
</p>

## 📊Result
### Qualitative Result
<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_vis.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>Visual comparisons with the manually crafted challenging prompts. We apply ToMeSD and SiTo on stable diffusion v1.5, achieving similar speed-up ratios of 1.63 and 1.65, respectively. Under these comparable conditions, our method generated more realistic, detailed images that better aligned with the original images and text prompts.</em>
</p>

### Quantitative Result
<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_result.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>Comparison between the proposed SiTo and ToMeSD with SD v1.5 and SD v2 on ImageNet and COCO30k.</em>
</p>

## 🛠 Usage
### Dependencies
To run SiTo for SD, PyTorch version `1.12.1` or higher is required (due to the use of `scatter_reduce`). You can download it from [here](https://pytorch.org/get-started/locally/).

###  Installation
```shell
git clone https://github.com/EvelynZhang-epiclab/SiTo.git
```
### Apply SiTo
Applying SiTo is very simple, you just need the following two steps (and no additional training is required):

Step1: Add our code package `sito` in the `scripts`.

Step2：Apply SiTo in SD v1 and SD v2:
Add the following code at the respective lines of [SD v1](https://github.com/runwayml/stable-diffusion/blob/08ab4d326c96854026c4eb3454cd3b02109ee982/scripts/txt2img.py#L241) or [SD v2](https://github.com/Stability-AI/stablediffusion/blob/fc1488421a2761937b9d54784194157882cbc3b1/scripts/txt2img.py#L220):

```python
import sito
sito.apply_patch(model,prune_ratio=0.5)
```
 As follows, we also provide more fine-grained parameter control.
```python
import sito
sito.apply_patch(model,
        prune_ratio = 0.7, # The pruning ratio
        max_downsample_ratio = 1, # The number of layers to prune in the Unet. It is recommended to prune only the first layer (see Fig. 7 in the paper for details)
        prune_selfattn_flag = True, # Whether to prune the self-attention layers. Recommended
        prune_crossattn_flag = False, # Whether to prune the cross-attention layers. Not recommended
        prune_mlp_flag: bool = False, # Whether to prune the MLP layers. Strongly not recommended (see Tab. 2 in the paper for details)
        sx: int = 2, sy: int = 2, # Patch size
        noise_alpha= 0.1, # Controls the noise level
        sim_beta:float = 1 
)
```
### Run SiTo

~~~python
# After setting up the environment, compile it.
pip install -v -e .
~~~

- Generate an image based on a prompt.
~~~
python scripts/txt2img.py --n_iter 1 --n_samples 1 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid --prompt "a photograph of an astronaut riding a horse"
~~~

- Read prompts from a `.txt` file to generate images. Use `--from-file imagenet.txt` for generating ImageNet 2k images, and `--from-file coco30k.txt` for generating COCO 30k images. 

~~~
python scripts/txt2img.py --n_iter 2 --n_samples 4 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid --from-file imagenet.txt
~~~

- When measuring speed, set `n_iter` to at least 2 (because at least one iteration is required for warm-up). Enable both `--skip_save` and `--skip_grid` to avoid saving images.

~~~
python scripts/txt2img.py --n_iter 3 --n_samples 8 --W 512 --H 512 --ddim_steps 50 --plms --skip_save --skip_grid --from-file xxx.txt
~~~
| Prompt File | Download Link | Extraction Code |
| --- | --- | --- |
| **imagenet.txt** | [Download imagenet.txt from Baidu](https://pan.baidu.com/s/19NQbMTmsCTCPz-NF2TjiWw) | `y1lo` |
| **coco30k.txt** | [Download coco30k.txt from Baidu](https://pan.baidu.com/s/1LveKxVASHzepwVwerdnhPA) | `jmhn` |

## 📐 Evaluation

### FID

This implementation references the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) repository.
Modify [this line](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L146C3-L146C64]) in  `pytorch_fid/fid_score.py` as follows:
~~~python
dataset = ImagePathDataset(files, transforms=my_transform)
my_transform = TF.Compose([
    TF.Resize((512, 512)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
~~~

- Download Data

| Dataset | Download Link | Extraction Code |
| --- | --- | --- |
| **coco30k.npz** | [Download COCO30K from Baidu](https://pan.baidu.com/s/1EtHNHH5CLGeubB3wl3qDwg) | `f061` |
| **imagenet50k.npz** | [Download Imagenet50K from Baidu](https://pan.baidu.com/s/1dkakRoaVWU3iWRrgSTsIvA) | `hsy2` |

~~~python
python -m pytorch_fid path/to/[datasets].npz path/to/images
~~~

### Time

- It is recommended to use torch.cuda.Event to measure time (since GPUs perform parallel computation, using the regular time.time can be inaccurate). Refer to the following code:

~~~py
time0= torch.cuda.Event(enable_timing=True)
time1= torch.cuda.Event(enable_timing=True)
time0.record()
# Place the code segment that needs to be measured for time here.
time1.record()
torch.cuda.synchronize() 
time_consume=time0.elapsed_time(time1)
~~~

_Note: When measuring speed, it is recommended to perform a warm-up (for example, exclude the time taken for the first few iterations from the statistics)._

## 💐 Acknowledgments

Special thanks to the creators of [ToMeSD](https://github.com/dbolya/tomesd) upon which this code is built, for their valuable work in advancing diffusion model acceleration.

## 🔗 Citation
If you use this codebase, or SiTo inspires your work, we would greatly appreciate it if you could star the repository and cite it using the following BibTeX entry.
```
@inproceedings{zhang2025training,
  title={Training-free and hardware-friendly acceleration for diffusion models via similarity-based token pruning},
  author={Zhang, Evelyn and Tang, Jiayi and Ning, Xuefei and Zhang, Linfeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={9},
  pages={9878--9886},
  year={2025}
}
```
## :e-mail: Contact
If you have more questions or are seeking collaboration, feel free to contact me via email at [`evelynzhang2002@163.com`](mailto:yuweizhang2002@gmail.com).
