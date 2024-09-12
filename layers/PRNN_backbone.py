from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import torch.nn.functional as F



def adjust_sequence_for_stride(x, seq_len, patch_len, stride):
    num_patches_without_padding = (seq_len - patch_len) // stride + 1    # 计算不进行padding时，可以切分出的patch数量
    last_patch_start = (num_patches_without_padding - 1) * stride        # 计算最后一个patch开始的位置
    total_pad_length = max(0, last_patch_start + patch_len - seq_len)    # 判断最后一个patch是否能够完整切分，并计算需要的padding长度
    x_padded = F.pad(x, (0, total_pad_length), 'constant', 0)            # 应用padding
    return x_padded



class PRNN_backbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):
        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.patch_len = patch_len
        self.stride = stride
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = target_window
        self.input_size = patch_len
        
        # RNN layer configuration
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layer to map RNN output to target window size
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        

    def forward(self, x):
        # norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        batch_size, c_in, seq_len = x.shape
        
        x_padded = adjust_sequence_for_stride(x, seq_len, self.patch_len, self.stride)
        patches = x_padded.unfold(2, self.patch_len, self.stride).contiguous()     # [batch_size, channel, num_patches, patch_len]
        patches = patches.view(patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3])  # [batch_size*channel, num_patches, patch_len]

        # Process patches with RNN
        rnn_out, _ = self.rnn(patches)  # [batch_size*channel, num_patches, hidden_size]

        # Use the output of the last patch for prediction
        last_patch_output = rnn_out[:, -1, :]   # [batch_size*channel, hidden_size]

        output = self.fc(last_patch_output)  # Shape: [batch_size*channel, output_size]

        # Reshape output to [batch_size, channel, target_window]
        output = output.view(batch_size, c_in, self.target_window)

        # denorm
        if self.revin:
            output = output.permute(0, 2, 1)
            output = self.revin_layer(output, 'denorm')
            output = output.permute(0, 2, 1)

        return output