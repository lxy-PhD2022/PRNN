from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *
import math


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, hidden_size, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = hidden_size*num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len)
        self.dropout = nn.Dropout(head_dropout)
        self.linear2 = nn.Linear(hidden_size, forecast_len)

    def forward(self, x):                     
        """
        x:      [bs x nvars x hidden_size x num_patch]
        output: [bs x target_dim x nvars]
        """
        x = self.flatten(x)     # x: [bs x nvars x (hidden_size * num_patch)]    
        x = self.dropout(x)
        x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]

        # last_patch_output = x[:, :, :, -1]   # [bs x nvars x hidden_size]
        # x = self.linear2(last_patch_output)  # [bs x nvars x forecast_len]
        # return x.transpose(2,1)              # [bs x forecast_len x nvars]
    

class PretrainHead(nn.Module):
    def __init__(self, hidden_size, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, patch_len)

    def forward(self, x):
        """
        x: tensor      [bs x nvars x hidden_size x num_patch]
        output: tensor [bs x num_patch x n_vars x patch_len]
        """
        x = x.transpose(3,2)                 # [bs x nvars x num_patch x hidden_size]
        x = self.linear(self.dropout(x))     # [bs x nvars x num_patch x patch_len]
        x = x.transpose(2,1)                 # [bs x num_patch x nvars x patch_len]
        return x


class RNN(nn.Module):
    def __init__(self, hidden_size:int, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.c_in = c_in
        self.target_window = target_dim
        self.hidden_size = hidden_size
        self.input_size = patch_len
        
        # RNN layer configuration
        self.rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)  # self.num_layers
        self.embed = nn.Linear(self.input_size, self.hidden_size)

        sr = num_patch
        ts = 1.0/sr
        t = np.arange(0,1,ts)
        t=torch.tensor(t).cuda()
        for i in range(num_patch//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)]) 
        self.cos = nn.Parameter(cos.float(), requires_grad=False)
        self.sin = nn.Parameter(sin.float(), requires_grad=False)
        
        # Fully connected layer to map RNN output to target window size
        if head_type == "pretrain":
            self.head = PretrainHead(hidden_size, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.c_in, hidden_size, num_patch, target_dim, head_dropout)
        

    def forward(self, x):
        batch_size, num_patch, c_in, patch_len = x.shape
        patches = x.transpose(1,2)                     # [bs x nvars x num_patch x patch_len]
        patches = patches.reshape(patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3])  # [bs*nvars x num_patch x patch_len]

        embedding = self.embed(patches) # [batch_size*channel, num_patch, hidden_size]
        rnn_out, _ = self.rnn(embedding)  # [batch_size*channel, num_patch, hidden_size]

        rnn_out = rnn_out.permute(0, 2, 1)  # [batch_size*channel, hidden_size, num_patch]
        frequency = torch.fft.rfft(rnn_out,axis=-1)  # 时间步做fft
        basis_cos = torch.einsum('bhk,kn->bhn', frequency.real, self.cos)   # [batch_size*channel, hidden_size, num_patch]
        basis_sin = torch.einsum('bhk,kn->bhn', frequency.imag, self.sin)   # [batch_size*channel, hidden_size, num_patch]
        rnn_out = basis_cos + basis_sin                                     # [batch_size*channel, hidden_size, num_patch]
        rnn_out = rnn_out.permute(0, 2, 1)  # [batch_size*channel, num_patch, hidden_size]
        
        second_rnn_out, _ = self.rnn(rnn_out)  # Shape: [batch_size*channel, num_patch, hidden_size]
        rnn_out = second_rnn_out

        rnn_out = rnn_out.view(batch_size, c_in, num_patch, -1)   # [bs x nvars x num_patch x hidden_size]  
        rnn_out = rnn_out.permute(0,1,3,2)                        # [bs x nvars x hidden_size x num_patch]
        output = self.head(rnn_out)                               # [bs x num_patch x nvars x patch_len]

        return output