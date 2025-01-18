import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Callable, Optional, Dict, Any
import math
from .utils import init_generator


def do_nothing(x: torch.Tensor, mode: str = None):
    return x

count_num = torch.zeros((2, 4096), dtype=torch.int32, device="cuda")


def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def find_patch_max_indices(tensor, sx, sy):  # tensor: (B,N)
    b, N = tensor.size()
    n = int(math.sqrt(N))
    tensor = tensor.view(b, n, n)
    h_patches = n // sy
    w_patches = n // sx
    tensor = tensor[:, :h_patches * sy, :w_patches * sx]
    tensor_reshaped = tensor.view(b, h_patches, sy, w_patches, sx)
    tensor_reshaped = tensor_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    tensor_reshaped = tensor_reshaped.view(b, h_patches, w_patches, sy * sx)
    _, max_indices = tensor_reshaped.max(dim=-1, keepdim=True)
    return max_indices
    
def duplicate_half_tensor(tensor):
    B, _, _,_ = tensor.shape
    if B % 2 != 0:
        raise ValueError("B must be even for this operation")
    first_half = tensor[:B // 2]
    tensor[B // 2:] = first_half
    return tensor

def prune_and_recover_tokens(metric: torch.Tensor,
                                 num_prune: int,
                                 w: int,
                                 h: int,
                                 sx: int,
                                 sy: int,
                                 sim_beta: float,
                                 noise_alpha: float,
                                 current_timestep: int
                                 ) -> Tuple[Callable, Callable]:
    B, N, C = metric.shape
    if num_prune <= 0:  # 如果r<0, 什么也不做
        return do_nothing, do_nothing
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    with torch.no_grad():
        metric = metric.to(torch.float16)
        metric = F.normalize(metric, p=2, dim=-1)
        consine_graph = torch.matmul(metric, metric.transpose(-1, -2))
        # my_norm = metric.norm(dim=-1, keepdim=True)
        # my_norm = my_norm.to(torch.float16)
        # metric = metric / my_norm
        # consine_graph = metric @ metric.transpose(-1, -2)
        hsy, wsx = h // sy, w // sx
        # ##############################################################
        # Method 1:   Select the highest score based on (sim_beta * SimScore + noise_alpha * Noise) within each patch.
        dst_score = consine_graph.sum(-1)
        noise_score= torch.randn(B, N, dtype=metric.dtype, device=metric.device)
        dst_score=sim_beta*dst_score+noise_alpha*noise_score
        rand_idx = find_patch_max_indices(dst_score, sx, sy)  # [B,hsy,wsx,1]
        # Align CFG
        # rand_idx = duplicate_half_tensor(rand_idx)
        ##############################################################
        # Method 2：LocalRandom
        # generator = init_generator(metric.device)
        # rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)  # [hsy, wsx, 1]
        # rand_idx = rand_idx.unsqueeze(0).expand(B, -1, -1, -1)
        #############################################################
        # Method3: Fixed selection of the top-left corner within each patch.
        # rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        # rand_idx = rand_idx.unsqueeze(0).expand(B,-1,-1,-1)
        ##############################################################
        idx_buffer_view = torch.zeros(B, hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=-1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view  # (B,h,w)
        else:
            idx_buffer = idx_buffer_view  # (B,h,w)
        rand_idx = idx_buffer.reshape(B, -1).argsort(dim=1)  # (B,N)
        del idx_buffer, idx_buffer_view

        #############################################################
        # Method 4: Randomly select within the global scope.
        # random_permutation = torch.randperm(N).to(metric.device)
        # rand_idx=random_permutation.unsqueeze(0).expand(B,-1)
        ############################################################
        # Method 5:  Select the maximum SimScore within the global scope.
        # dst_score = consine_graph.sum(-1) # (B,N)
        # _, rand_idx = torch.sort(dst_score, dim=1, descending=True)
        #############################################################
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:]  # src  (B,N-num_dst) 0~N
        b_idx = rand_idx[:, :num_dst]  # dst  (B,num_dst) 0~N

        score_sim_1 = gather(consine_graph, dim=1, index=a_idx.unsqueeze(-1).expand(B, -1, N))  # (B,N-num_dst,N)
        score_sim = gather(score_sim_1, dim=-1,
                           index=b_idx.unsqueeze(1).expand(B, score_sim_1.shape[1], -1))  # (B,N-num_dst,num_dst)
        score_sim_value, score_sim_index = torch.max(score_sim, dim=2)  # (B,N-num_dst) 0~num_dst
        total_score = - score_sim_value

        edge_idx = total_score.argsort(dim=-1)  # The size is (B, N-num_dst), sorted in ascending order, with values as indices (0~N-num_dst).
        unm_idx = edge_idx[..., num_prune:]  # Size: (B,N-num_dst-num_prune) Value:（0~N-num_dst） [8,1024]  0~3071
        src_idx = edge_idx[..., :num_prune]  # Size: (B,num_prune) Value:（0~N-num_dst）           [8,2048]  0~3071    1896
        a_idx_tmp = a_idx.expand(B, N - num_dst)  # (8,3072) 0~4095
        a_unm_idx = gather(a_idx_tmp, dim=1, index=unm_idx)  # Size: (B,N-num_dst-unm_prop) Value:（0~N）
        a_src_idx = gather(a_idx_tmp, dim=1,
                           index=src_idx)  # Size:(B,num_prune) Value（0~N）  idx 1896 out of bound with size 1024
        #################################################################################
        # Count and visualize the number of pruning occurrences.
        """
        global count_num
        count_num.scatter_add_(1, a_src_idx, torch.ones_like(a_src_idx, dtype=torch.int32))

        if current_timestep ==1:
            count_num.to("cpu")
            n = int(np.sqrt(N))

            max_value = count_num.max().item()
            #print('max_value',max_value)
            
            value_range = torch.arange(1, max_value + 1)  
            histograms = torch.zeros((B, max_value + 1), dtype=torch.int32)

            for b in range(B):
                histograms[b] = torch.bincount(count_num[b], minlength=max_value + 1)

            for b in range(B):
                non_zero_mask = histograms[b][1:] > 0  
                non_zero_values = value_range[non_zero_mask]
                non_zero_counts = histograms[b][1:][non_zero_mask]

                plt.figure(figsize=(6, 6))
                plt.bar(non_zero_values.numpy(), non_zero_counts.numpy(),color='#b5e48c')
                plt.xlim(0, 250)  
                
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                # plt.xlabel('Value')
                # plt.ylabel('Count')
                # plt.title(f'Distribution of Values for B = {b} (non-zero counts)')
                if b==1:
                    plt.savefig(f'/root/autodl-tmp/stable-diffusion-main/outputs/txt2img-samples/dst_1/test/samples/with_noise_prunenum.svg',format='svg')  # 保存图片
                    plt.close()  

            
            for b in range(B):
                plt.figure(figsize=(6, 6))
                count_matrix = count_num[b].reshape(n, n).cpu().numpy()

                heatmap = plt.imshow(count_matrix, cmap='summer', interpolation='nearest', vmin=0, vmax=250)
                cbar = plt.colorbar(heatmap)

                # Adjust the layout to ensure the colorbar and heatmap align properly
                plt.subplots_adjust(left=0.1, right=0.8, top=0.6, bottom=0.2)

                if b == 1:
                    plt.savefig(f'/root/autodl-tmp/stable-diffusion-main/outputs/txt2img-samples/dst_1/test/samples/with_noise_heatmap.svg', format='svg')
                plt.close()
        """
        ###################################################################################################
        combined = torch.cat((a_unm_idx, b_idx), dim=1)  # (B,N-num_prune)
        weight = gather(consine_graph, dim=1, index=combined.unsqueeze(-1).expand(B, -1, N))  # (B,N-num_prune,N)
        weight_prop = gather(weight, dim=2,
                             index=a_src_idx.unsqueeze(1).expand(-1, weight.shape[1], -1))  # (B,N-prune,num_prune)
        _, max_indices = torch.max(weight_prop, dim=1)  # (B,num_prune) 0~N-num_prop

    def prune_tokens(x: torch.Tensor) -> torch.Tensor:  # x: (B,N,C)
        B, N, C = x.shape  #
        unm = gather(x, dim=-2,
                     index=a_unm_idx.unsqueeze(2).expand(B, N - num_dst - num_prune, C))  # (B, N-num_dst-num_prune, C)
        dst = gather(x, dim=-2, index=b_idx.unsqueeze(2).expand(B, num_dst, C))  # (B,num_dst,C)
        result = torch.cat([unm, dst], dim=1)
        return result  # (B,N-num_prune,c)

    def recover_tokens(x: torch.Tensor) -> torch.Tensor:  # (B,N-num_prune,c)
        unm_len = a_unm_idx.shape[1]  # N-num_dst-num_prune
        # unm: (B, N-num_dst-num_prune,C) dst: (B,num_dst,C)
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape
        # weight_prop: (B,num_prune,num_dst)
        # dst: (B,num_dst,C)
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        src = torch.gather(x, 1, max_indices.unsqueeze(-1).expand(-1, -1, c))  # (B,num_prune,c)
        out.scatter_(dim=-2, index=b_idx.unsqueeze(2).expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=a_unm_idx.unsqueeze(2).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=a_src_idx.unsqueeze(2).expand(B, num_prune, c), src=src)
        return out  # (B,N,C)

    return prune_tokens, recover_tokens

