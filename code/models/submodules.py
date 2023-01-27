import math
import torch
from torch import nn
from copy import deepcopy
from einops import rearrange
from torch.nn import functional as tf


class PositionalEncoder(nn.Module):
    """
        Positional Encoder from the original paper "Attention is all you need"
    """
    def __init__(self, input_size: int, max_seq_len: int=160):
        """
        input_size: encoder's dimension, i.e. time series embedding vector size
        max_seq_len: the maximum time step
        """
        super().__init__()
        assert input_size % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.input_size = input_size
        pe = torch.zeros(max_seq_len, input_size)
        for pos in range(max_seq_len):
            for i in range(0, input_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / input_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / input_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        x: [Batch, input_size]
        """
        with torch.no_grad():
            x = x * math.sqrt(self.input_size)
            return x + self.pe[:, :x.size(1)]


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, 
        stride: int=1, dilation: int=1, groups: int=1, bias: int=True
    ):
        """
        in_channels: input's channel
        out_channels: output's channel
        kernel_size: size of the convolving kernel
        stride: stride of the convolution
        dilation: spacing between kernel elements
        groups: number of blocked connections from input channels to output channels
        bias: if True, adds a learnable bias to the output.
        """
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias
        )
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, x: torch.Tensor):
        """
        x: [Batch, input_size]
        """
        return super(CausalConv1d, self).forward(tf.pad(x, (self.__padding, 0)))
    
    
class ScaledDotProductAttention(nn.Module):
    """
        Scaled Dot Product Self Attention from the original paper "Attention is all you need"
    """
    def __init__(self, dropout: float=None, scale: bool=True):
        """
        dropout: dropout probability
        scale: if True, the attention product would be scaled based on the input dimension
        """
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None):
        """
        q: [Batch, T, input_size]
        k: [Batch, T, input_size]
        v: [Batch, T, input_size]
        mask: [T, input_size]
        """
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn
    

class DCSA(nn.Module):
    """
        Dual Convolutional Self-Attention
    """
    def __init__(
        self, input_size: int, kernel_size: int, self_attn: nn.Module, hidden_size: int, dropout: float, second_input: bool=True
    ):
        """
        input_size: embedding dimension
        kernel_siz: size of the convolving kernel
        self_attn: self attention neural network
        hidden_size: size of the hidden dimension in the GRN 
        dropout: dropout probability
        second_input: if True, a second input (from different modality) will be added
        """
        super().__init__()
        
        self.self_attn = self_attn
        
        self.norm1 = nn.LayerNorm(input_size)
        self.causal_k1 = CausalConv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)
        self.causal_q1 = CausalConv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)
        if second_input:
            self.norm2 = nn.LayerNorm(input_size)        
            self.causal_k2 = CausalConv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)
            self.causal_q2 = CausalConv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)

        self.ffn = GatedResidualNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=input_size,
            dropout=dropout, context_size=input_size
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor=None, mask: torch.Tensor=None):
        """
        x1: [Batch, T, input_size]
        x2: [Batch, T, input_size]
        mask: [T, input_size]
        """
        residual1 = x1
        k1 = tf.tanh(self.causal_k1(x1.permute(0, 2, 1)))
        q1 = tf.tanh(self.causal_q1(x1.permute(0, 2, 1)))
        x1 = self.self_attn(q1.permute(0, 2, 1), k1.permute(0, 2, 1), x1, mask)[0]
        x1 = self.norm1(x1 + residual1)
        
        if x2 is not None:
            residual2 = x2
            k2 = tf.tanh(self.causal_k2(x2.permute(0, 2, 1)))
            q2 = tf.tanh(self.causal_q2(x2.permute(0, 2, 1)))
            x2 = self.self_attn(q2.permute(0, 2, 1), k2.permute(0, 2, 1), x2, mask)[0]
            x2 = self.norm2(x2 + residual2)
            return self.ffn(x1, x2)
        
        return self.ffn(x1)
    

class MDCSA(nn.Module):
    """
        Multihead Dual Convolutional Self-Attention
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        """
        input_size : Embedding Dimensions.
        hidden_size : Size of the hidden dimension in the GRN inside DCSA.
        kernel_sizes : List of kernel size for each DCSA
        dropout : Dropout probability.
        """
        super().__init__()

        self_attn = ScaledDotProductAttention(dropout)
        self.sa = deepcopy(self_attn)
        self.norm = nn.LayerNorm(input_size)
        self.dcsa1 = DCSA(input_size, 1, deepcopy(self_attn), hidden_size, dropout)
        self.dcsa4 = DCSA(input_size, 4, deepcopy(self_attn), hidden_size, dropout)
        self.dcsa7 = DCSA(input_size, 7, deepcopy(self_attn), hidden_size, dropout)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, stride=3)

    def forward(self, x1, x2, mask):
        """
        x1: [Batch, T, input_size]
        x2: [Batch, T, input_size]
        """
        _x1 = self.dcsa1(x1, x2, mask)
        _x4 = self.dcsa4(x1, x2, mask)
        _x7 = self.dcsa7(x1, x2, mask)
        x = torch.stack([_x1, _x4, _x7], dim=1)
        x = rearrange(x, 'b (n1 n2) (l1 l2) d -> b (n1 l1) (n2 l2) d', n1=1, l1=x1.shape[1])
        x = rearrange(x, 'b l n d -> b (l n) d')
        x = self.sa(x, x, x, None)[0]
        x = self.conv1d(x.permute(0, 2, 1))       
        return self.norm(x.permute(0, 2, 1))
    
    
class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit based on 'Language modeling with gated convolutional networks'
    Code is from https://github.com/jdb78/pytorch-forecasting
    """

    def __init__(self, input_size: int, hidden_size: int=None, dropout: float=None):
        """
        input_size: input dimension
        hidden_size: hidden dimension
        dropout: dropout probability
        """
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor):
        """
        x: [Batch, input_size]
        """
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = tf.glu(x, dim=-1)
        return x
    
    
class TimeDistributedInterpolation(nn.Module):
    """
    Layer to resize input size to output size in a timely distributed manner
    Code is from https://github.com/jdb78/pytorch-forecasting
    """
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        """
        output_size: target output dimension
        batch_first: whether batch dimension is in first or second 
        trainable: whether this layer can be trained by having parameters
        """
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x: torch.Tensor):
        upsampled = tf.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x: torch.Tensor):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        # x_reshape: [Batch * T, input_size]
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # y: [Batch, T, output_size]
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  
        else:
            # [T, Batch, output_size]
            y = y.view(-1, x.size(1), y.size(-1))

        return y
    

class ResampleNorm(nn.Module):
    """
    Layer to resize and normalize input
    Code is from https://github.com/jdb78/pytorch-forecasting
    """
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        """
        input_size: input dimension
        output_size: output dimension
        trainable_add: simple non-linear transformation to input
        """
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor):
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output

    
class AddNorm(nn.Module):
    """
    Adding and Normalisation Layer with trainable parameters
    Code is from https://github.com/jdb78/pytorch-forecasting
    """
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        """
        input_size: input dimension
        skip_size: input dimension that does not go through transformation
        trainable_add: whether addition part is trainable through parameters and activation function
        """
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output
    

class GateAddNorm(nn.Module):
    """
    GLU with AddNorm layer
    Code is from https://github.com/jdb78/pytorch-forecasting
    """
    def __init__(
        self, input_size: int, hidden_size: int = None, skip_size: int = None,
        trainable_add: bool = False, dropout: float = None,
    ):
        """
        input_size: input dimension
        hidden_size: hidden dimension
        skip_size: skip input dimension (in case skip input has different dimension than input_size)
        trainable_add: whether adding has trainable parameters with activation function
        dropout: dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output

    
class GatedResidualNetwork(nn.Module):
    """
    Gated residual network based on 'Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting'
    Code is from https://github.com/jdb78/pytorch-forecasting
    """
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, 
        dropout: float = 0.1, context_size: int = None, residual: bool = False,
    ):
        """
        input_size: input dimension
        hidden_size: hidden dimension
        output_size: output dimension
        dropout: dropout layer probability
        context_size: extra context dimension
        residual: whether the residual connection is from input_size or output_size
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, context: torch.Tensor=None, residual: torch.Tensor=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x