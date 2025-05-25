########################################################################################################## 频域压缩+维纳辛钦
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


class FFT_PRNN_backbone(nn.Module):
    def __init__(self, k: int, hidden_size: int, num_layers: int, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
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
        self.k = k
        
        # RNN layer configuration
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)  # self.num_layers
        self.embed = nn.Linear(self.context_window, self.hidden_size)
        
        # for frequency compression
        sr = context_window
        ts = 1.0/sr
        t = np.arange(0,1,ts)
        t = torch.tensor(t).cuda()
        for i in range(context_window//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)]) 
        self.cos = nn.Parameter(cos, requires_grad=False)
        self.sin = nn.Parameter(sin, requires_grad=False)
        
        # for weinaxinqin 
        sr2 = k
        ts2 = 1.0/sr2
        t2 = np.arange(0,1,ts2)
        t2 = torch.tensor(t2).cuda()
        for i2 in range(k//2+1):
            if i2==0:
                cos2=0.5*torch.cos(2*math.pi*i*t2).unsqueeze(0)
                sin2=-0.5*torch.sin(2*math.pi*i*t2).unsqueeze(0)
            else:
                cos2=torch.vstack([cos2,torch.cos(2*math.pi*i*t2).unsqueeze(0)])
                sin2=torch.vstack([sin2,-torch.sin(2*math.pi*i*t2).unsqueeze(0)]) 
        self.cos2 = nn.Parameter(cos2, requires_grad=False)
        self.sin2 = nn.Parameter(sin2, requires_grad=False)
        
        # Fully connected layer to map RNN output to target window size
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        

    def forward(self, x):
        # norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        batch_size, c_in, seq_len = x.shape
        frequency = torch.fft.rfft(x,axis=-1)  # 时间步做fft
        basis_cos = torch.einsum('bhk,kn->bhkn', frequency.real, self.cos)   # [b, c, l/2+1, l]
        basis_sin = torch.einsum('bhk,kn->bhkn', frequency.imag, self.sin)   # [b, c, l/2+1, l]
        x = basis_cos + basis_sin   
        # top k频域压缩
        power = (frequency.real**2 + frequency.imag**2)       # 计算功率谱，等价于torch.abs(x)**2)
        topk_power, topk_indices = torch.topk(power, k=self.k, dim=2)         # 沿频率维度取top-k, [b, c, k]
        x = torch.gather(x, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))) # [b, c, k, l]    
        # 第一遍RNN
        embedding = self.embed(x.reshape(batch_size*c_in, self.k, seq_len)) # [bc, k, h]
        rnn_out, _ = self.rnn(embedding)   # [bc, k, h]
        
        # 将隐状态的top k频率输入到第二遍RNN  
        rnn_out_frequency = torch.fft.rfft(rnn_out.permute(0, 2, 1), axis=-1) # 时间步做fft, [bc, h, k/2+1]
        basis_cos2 = torch.einsum('bhk,kn->bhn', rnn_out_frequency.real, self.cos2)   # [bc, h, k]
        basis_sin2 = torch.einsum('bhk,kn->bhn', rnn_out_frequency.imag, self.sin2)   # [bc, h, k]
        x = basis_cos2 + basis_sin2                                                   # [bc, h, k]
        
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))                              # [bc, k, h]  

        # Use the output of the last patch for prediction
        last_patch_output = rnn_out[:, -1, :]   # [batch_size*channel, hidden_size]
        output = self.fc(last_patch_output)  # Shape: [batch_size*channel, output_size]
        output = output.view(batch_size, c_in, self.target_window)

        # denorm
        if self.revin:
            output = output.permute(0, 2, 1)
            output = self.revin_layer(output, 'denorm')
            output = output.permute(0, 2, 1)

        return output