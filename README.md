# SiTo: Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning (AAAI-2025)
📰 This is the official code for our paper: [《Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning》](https://www.researchgate.net/publication/387204421_Training-Free_and_Hardware-Friendly_Acceleration_for_Diffusion_Models_via_Similarity-based_Token_Pruning)
## 🔥 News
- `2024/12/10`🤗🤗 SiTo is accepted by AAAI-2025
- `2025/1/18` 💥💥 We release the code for our work [SiTo](https://github.com/EvelynZhang-epiclab/SiTo) about accelerating diffusion models for FREE. 🎉 **The zero-shot evaluation shows SiTo leads to 1.90x and 1.75x acceleration on COCO30K and ImageNet with 1.33 and 1.15 FID reduction at the same time. Besides, SiTo has no training requirements and does not require any calibration data, making it plug-and-play in real-world applications.**

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

