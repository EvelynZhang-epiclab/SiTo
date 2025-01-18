# SiTo: Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning (AAAI-2025)
<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_logo.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:50%">
  <br>
</p>

📰 This is the official code for our paper: [《Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning》](https://www.researchgate.net/publication/387204421_Training-Free_and_Hardware-Friendly_Acceleration_for_Diffusion_Models_via_Similarity-based_Token_Pruning)
## 🔥 News
- `2024/12/10`🤗🤗 SiTo is accepted by AAAI-2025
- `2025/1/18` 💥💥 We release the code for our work [SiTo](https://github.com/EvelynZhang-epiclab/SiTo) about accelerating diffusion models for FREE. 🎉 **The zero-shot evaluation shows SiTo leads to 1.90x and 1.75x acceleration on COCO30K and ImageNet with 1.33 and 1.15 FID reduction at the same time. Besides, SiTo has no training requirements and does not require any calibration data, making it plug-and-play in real-world applications.**
## 🚀Overview
### Method
SiTo has a three-stage pipeline. 
- SiTo carefully selects a set of **base tokens** which are utilized as the base to select and recover the pruned tokens.
- SiTo selects the tokens that have the highest similarity to the base tokens as the **pruned tokens**.
- SiTo feeds the unpruned tokens to the neural layers and **recovers the pruned tokens** by directly copying their most similar base tokens.

<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_overview.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>The pipeline of SiTo on the example of self-attention. (a) Base Token Selection: We compute the Cosine Similarity between all the tokens. For each token, we sum its similarity to all the tokens as the SimScore. Then, Gaussian Noise is added to the SimScore introduces randomness, preventing identical base and pruned token choices across timesteps. Finally, the token that has the highest Noise SimScore in an image patch is selected as a base token. (b) Pruned Token Selection: The tokens with the highest similarity to the base tokens are selected as pruned tokens. (c) Pruned Token Recovery: The unpruned tokens are fed to the neural layers. Then, the pruned tokens are recovered by copying from their most similar base tokens.</em>
</p>

### 👀 Qualitative Result
<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_vis.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>Visual comparisons with the manually crafted challenging prompts. We apply ToMeSD and SiTo on stable diffusion v1.5, achieving similar speed-up ratios of 1.63 and 1.65, respectively. Under these comparable conditions, our method generated more realistic, detailed images that better aligned with the original images and text prompts.</em>
</p>

### 📊 Quantitative Result
<p align="center">
  <img src="https://github.com/EvelynZhang-epiclab/EvelynImgs/blob/main/sito_result.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:88%">
  <br>
  <em>Comparison between the proposed SiTo and ToMeSD with SD v1.5 and SD v2 on ImageNet and COCO30k.</em>
</p>

## 🛠 Usage
Applying SiTo is very simple, you just need the following two steps (and no additional training is required):

1. Add our code package `sitosd` in the scripts.

2. Apply SiTo in SD v1 and SD v2:

SD1: https://github.com/runwayml/stable-diffusion/blob/08ab4d326c96854026c4eb3454cd3b02109ee982/scripts/txt2img.py#L241

SD2: https://github.com/Stability-AI/stablediffusion/blob/fc1488421a2761937b9d54784194157882cbc3b1/scripts/txt2img.py#L220

Add the following code at the respective lines:

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

## 💐 Acknowledgments

Special thanks to the creators of [ToMeSD](https://github.com/dbolya/tomesd) upon which this code is built, for their valuable work in advancing diffusion model acceleration.

## 🔗 Citation
If you use this codebase, or SiTo inspires your work, we would greatly appreciate it if you could star the repository and cite it using the following BibTeX entry.
```
@inproceedings{zhang2025sito,
  title={Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning},
  author={Zhang, Evelyn and Tang, Jiayi and Ning, Xuefei and Zhang, Linfeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={},
  number={},
  pages={},
  year={2025}
}
```
## :e-mail: Contact
If you have more questions or are seeking collaboration, feel free to contact me via email at [`yuweizhang2002@gmail.com`](mailto:yuweizhang2002@gmail.com).
