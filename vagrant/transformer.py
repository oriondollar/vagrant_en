import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Trans(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, p_dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, p_dropout)
        self.cross_attn = MultiHeadedAttention(n_heads, d_model, p_dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.sublayer = clones(SublayerConnection(d_model, p_dropout), 3)

    def forward(self, y, z, mask):
        y = self.sublayer[0](y, lambda y: self.self_attn(query=y, key=y, value=y, mask=mask))
        y = self.sublayer[1](y, lambda y: self.cross_attn(query=y, key=z, value=z, mask=mask))
        return self.sublayer[2](y, self.ff)

class Predictor(nn.Module):
    def __init__(self, d_latent, width, depth, act_fn):
        super().__init__()
        pred_layers = []
        for i in range(depth):
            if i == 0:
                pred_layers.append(nn.Linear(d_latent, width))
                pred_layers.append(act_fn)
            elif i == depth - 1:
                pred_layers.append(nn.Linear(width, 1))
            else:
                pred_layers.append(nn.Linear(width, width))
                pred_layers.append(act_fn)
        self.predict = nn.Sequential(*pred_layers)

    def forward(self, z):
        return self.predict(z)

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        ### Linear projections to get queries, keys and values
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        ### Apply attention
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        return x

class Deconv(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        self.reshape = nn.Sequential(
                nn.Linear(d_latent, 576),
                nn.SiLU())

        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (d_model - in_d) // 4 + in_d
            stride = 4 - i
            kernel_size = 11
            if i == 2:
                out_d = d_model
                stride = 1
            deconv_layers.append(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                    stride=stride, padding=2))
            deconv_layers.append(nn.SiLU())
            in_d = out_d
        self.deconv = nn.Sequential(*deconv_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, z):
        z = self.reshape(z).view(-1, 64, 9)
        z = self.deconv(z).permute(0, 2, 1)
        z = self.norm(z)[:,:-1,:]
        return z

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.silu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
