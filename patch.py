import torch
import math
from typing import Type, Dict, Any, Tuple, Callable
import copy
from . import sito
from .utils import isinstance_str, init_generator

import torch.nn.functional as F
import time

def select_sito_method(x: torch.Tensor, sito_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    args = sito_info["args"]
    current_timestep = sito_info["timestep"]
    original_h, original_w = sito_info["size"]  # 64,64
    original_tokens = original_h * original_w
    downsaple_ratio = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    if (downsaple_ratio <= args["max_downsample_ratio"]):  
        w = int(math.ceil(original_w / downsaple_ratio))
        h = int(math.ceil(original_h / downsaple_ratio))
        num_prune = int(x.shape[1] * args["prune_ratio"])
        p, r = sito.prune_and_recover_tokens(x, num_prune, w=w, h=h, sx=args['sx'], sy=args['sy'],noise_alpha=args['noise_alpha'],sim_beta=args['sim_beta'],current_timestep=current_timestep)

    else:
        p, r = (sito.do_nothing, sito.do_nothing)
    p_a, r_a = (p, r) if args["prune_selfattn_flag"] else (sito.do_nothing, sito.do_nothing)
    p_c, r_c = (p, r) if args["prune_crossattn_flag"] else (sito.do_nothing, sito.do_nothing)
    p_m, r_m = (p, r) if args["prune_mlp_flag"] else (sito.do_nothing, sito.do_nothing)
    return p_a, p_c, p_m, r_a, r_c, r_m


def make_sito_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class sitoBlock(block_class):
        _parent = block_class
        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            p_a, p_c, p_m, r_a, r_c, r_m = select_sito_method(x, self._sito_info)  # 选择方法
            # self-attention
            prune_x = p_a(self.norm1(x))
            out1 = self.attn1(prune_x, context=context if self.disable_self_attn else None)
            x = r_a(out1) + x
            # cross-attention
            prop_x = p_c(self.norm2(x))
            out2= self.attn2(prop_x, context=context)
            x = r_c(out2) + x
            # MLP
            x = r_m(self.ff(p_m(self.norm3(x)))) + x
            return x
    return sitoBlock


def make_diffusers_sito_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """

    class sitoBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=None,
                cross_attention_kwargs=None,
                class_labels=None,
        ) -> torch.Tensor:
            # (1) sito
            p_a, p_c, p_m, r_a, r_c, r_m = select_sito_method(hidden_states, self._sito_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) sito p_a
            norm_hidden_states = p_a(norm_hidden_states)
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) sito r_a
            hidden_states = r_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) sito p_c
                norm_hidden_states = p_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) sito r_c
                hidden_states = r_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) sito p_m
            norm_hidden_states = p_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) sito r_m
            hidden_states = r_m(ff_output) + hidden_states

            return hidden_states

    return sitoBlock


def hook_sito_model(model: torch.nn.Module):
    def hook(module, args):
        module._sito_info["size"] = (args[0].shape[2], args[0].shape[3])
        module._sito_info["timestep"] = args[1][0].cpu().item()
        return None
    model._sito_info["hooks"].append(model.register_forward_pre_hook(hook))

def apply_patch(
        model: torch.nn.Module,
        prune_ratio: int = 0.5,
        max_downsample_ratio: int = 1,
        prune_selfattn_flag: bool = True,
        prune_crossattn_flag: bool = False,
        prune_mlp_flag: bool = False,
        sx: int = 2, sy: int = 2,
        noise_alpha:float = 0.1,
        sim_beta:float = 1
):
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._sito_info = {
        "name": None,
        "size": None,
        "timestep": None,
        "hooks": [],
        "args": {
            "prune_selfattn_flag": prune_selfattn_flag,
            "prune_crossattn_flag": prune_crossattn_flag,
            "prune_mlp_flag": prune_mlp_flag,
            "prune_ratio": prune_ratio,
            "max_downsample_ratio": max_downsample_ratio,
            "sx": sx, "sy": sy,
            "noise_alpha":noise_alpha,
            "sim_beta":sim_beta
        }
    }
    hook_sito_model(diffusion_model)  # 添加size属性
    for x, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_sito_block_fn = make_diffusers_sito_block if is_diffusers else make_sito_block
            module.__class__ = make_sito_block_fn(
                module.__class__)
            module._sito_info = diffusion_model._sito_info
            module._myname = x
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
    return model

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a sito Diffusion module if it was already patched. """
    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if hasattr(module, "_sito_info"):
            for hook in module._sito_info["hooks"]:
                hook.remove()
            module._sito_info["hooks"].clear()

        if module.__class__.__name__ == "sitoBlock":
            module.__class__ = module._parent

    return model

'''
Unet Name
input_blocks.1.1.transformer_blocks.0
input_blocks.2.1.transformer_blocks.0

input_blocks.4.1.transformer_blocks.0
input_blocks.5.1.transformer_blocks.0

input_blocks.7.1.transformer_blocks.0
input_blocks.8.1.transformer_blocks.0

middle_block.1.transformer_blocks.0

output_blocks.3.1.transformer_blocks.0
output_blocks.4.1.transformer_blocks.0
output_blocks.5.1.transformer_blocks.0

output_blocks.6.1.transformer_blocks.0
output_blocks.7.1.transformer_blocks.0
output_blocks.8.1.transformer_blocks.0

output_blocks.9.1.transformer_blocks.0
output_blocks.10.1.transformer_blocks.0
output_blocks.11.1.transformer_blocks.0
'''