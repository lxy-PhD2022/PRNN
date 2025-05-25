from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import torch.nn.functional as F
import math



def adjust_sequence_for_stride(x, seq_len, patch_len, stride):
    num_patches_without_padding = (seq_len - patch_len) // stride + 1    
    last_patch_start = (num_patches_without_padding - 1) * stride        
    total_pad_length = max(0, last_patch_start + patch_len - seq_len)    
    x_padded = F.pad(x, (0, total_pad_length), 'constant', 0)            
    return x_padded



class FFT_PRNN_backbone(nn.Module):
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
        self.rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)  # self.num_layers
        self.embed = nn.Linear(self.input_size, self.hidden_size)
        
        num_patches_without_padding = (context_window - patch_len) // stride + 1    
        last_patch_start = (num_patches_without_padding - 1) * stride               
        total_pad_length = max(0, last_patch_start + patch_len - context_window)    
        padding_len = context_window + total_pad_length
        num_patches = (padding_len - patch_len) // stride + 1                       
        
        sr = num_patches
        ts = 1.0/sr
        t = np.arange(0,1,ts)
        t=torch.tensor(t).cuda()
        for i in range(num_patches//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)]) 
        self.cos = nn.Parameter(cos, requires_grad=False)
        self.sin = nn.Parameter(sin, requires_grad=False)
        
        # Fully connected layer to map RNN output to target window size
        self.fc = nn.Linear(self.hidden_size*num_patches, self.output_size)
        

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

        # uniform hidden size for the first and second learning of RNN
        embedding = self.embed(patches) # [batch_size*channel, num_patches, hidden_size]
        # Process patches with RNN
        rnn_out, _ = self.rnn(embedding)  # [batch_size*channel, num_patches, hidden_size]
        # time-frequency encoding
        rnn_out = rnn_out.permute(0, 2, 1)  # [batch_size*channel, hidden_size, num_patches]
        frequency = torch.fft.rfft(rnn_out,axis=-1)  # fft along time dimension
        basis_cos = torch.einsum('bhk,kn->bhn', frequency.real, self.cos)   # [batch_size*channel, hidden_size, num_patches]
        basis_sin = torch.einsum('bhk,kn->bhn', frequency.imag, self.sin)   # [batch_size*channel, hidden_size, num_patches]
        rnn_out = basis_cos + basis_sin                                     # [batch_size*channel, hidden_size, num_patches]
        rnn_out = rnn_out.permute(0, 2, 1)  # [batch_size*channel, num_patches, hidden_size]
        
        # input power spectrum into RNN          
        second_rnn_out, _ = self.rnn(rnn_out)  # Shape: [batch_size*channel, num_patches, hidden_size]
        rnn_out = second_rnn_out

        # Use all hidden states for prediction
        last_patch_output = rnn_out.reshape(batch_size*c_in, -1)   # [batch_size*channel, hidden_size]
        output = self.fc(last_patch_output)  # Shape: [batch_size*channel, output_size]
        # Reshape output to [batch_size, channel, target_window]
        output = output.view(batch_size, c_in, self.target_window)

        # denorm
        if self.revin:
            output = output.permute(0, 2, 1)
            output = self.revin_layer(output, 'denorm')
            output = output.permute(0, 2, 1)

        return output
