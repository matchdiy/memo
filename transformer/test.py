import torch
from torch import LongTensor, norm
from torch.nn import Embedding
import math
import numpy as np
import pandas as pd
import altair as alt
import copy


def test_embedding(max_norm):
  sentences = LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
  embedding = Embedding(num_embeddings=10, embedding_dim=4, max_norm=max_norm)
  print("-" * 64)
  print("[Embedding Weight]")
  print("-" * 64)
  print(embedding.weight)

  sentence = sentences[0]
  print("-" * 32)
  print("index tensor")
  print("-" * 32)
  print(sentence)
  print("\n")

  print("-" * 64)
  print("sentence embedding vectors")
  print("-" * 64)
  embed_vect = embedding(sentence)


  print(embed_vect)
  print("-" * 64)
  print("[Embedding Weight]")
  print("-" * 64)
  print(embedding.weight)

class PositionalEncoding(torch.nn.Module):
    '''
    Implement the PE function. usage:
      pe = PositionalEncoding(d_model, 0.1, msl)
      x = torch.zeros(mini_batch, msl, d_model)
      y = pe.forward(x)
    '''

    def __init__(self, d_model, dropout, max_len=5120):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def clones(module, N):
  "Produce N identical layers."
  return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
  "Compute 'Scaled Dot Product Attention'"
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = scores.softmax(dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    "Take in model size and number of heads."
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = torch.nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    "Implements Figure 2"
    if mask is not None:
        # Same mask applied to all h heads.
        mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    for lin, x in zip(self.linears, (query, key, value)):
      q = lin(x)
      print('{0} = dot({1}, {2})'.format(q.shape, x.shape, lin))
      q1 = q.view(nbatches, -1, self.h, self.d_k)
      q2 = q1.transpose(1, 2)
      print('view {}'.format(q1.shape))
      print('tran {}'.format(q2.shape))

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = [
        lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for lin, x in zip(self.linears, (query, key, value))
    ]

    # 2) Apply attention on all the projected vectors in batch.
    x, self.attn = attention(
        query, key, value, mask=mask, dropout=self.dropout
    )

    # 3) "Concat" using a view and apply a final linear.
    x = (
        x.transpose(1, 2)
        .contiguous()
        .view(nbatches, -1, self.h * self.d_k)
    )
    del query
    del key
    del value
    return self.linears[-1](x)

class Layer_norm(torch.nn.Module):
  def __init__(self, eps=1e-5):
    super(Layer_norm, self).__init__()
    self.eps = eps
    
  def forward(self, x):
    mean = torch.mean(x, dim=(1, 2, 3), keepdim=False)
    var = torch.var(x, dim=(1, 2, 3), keepdim=False)
    std = torch.sqrt(var + self.eps)
    return (x - mean[:,None,None,None]) / std[:,None,None,None]



def l1_forward_n(size, m, k, pp, bpe):
  lhs_k = 64
  n = (size - m*lhs_k*bpe*pp)/(m*bpe + k*bpe*pp)
  print("L1-Forward: size={0}KB, m={1}, k={2}, bpe={3}, pp={4}: n={5}".format(
    size/1024, m, k, bpe, pp, n))

def l1_forward(size, lhs, rhs, out, bpe, with_head=False):
  lhs_size = lhs[0] * lhs[1] * bpe
  rhs_size = rhs[0] * rhs[1] * bpe
  out_size = out[0] * out[1] * bpe
  used = lhs_size*lhs[2] + rhs_size*rhs[2] + out_size*out[2]
  utili = 100.0 * used / size
  
  # print("{:>4d}x{:_<4d}_{:_>4d}x{:_<4d}_{:_>4d}x{:_<4d}_{}x{}x{}, {:6.1f}KB({:6.1f}KB), {:6.1f}KB({:6.1f}KB), {:6.1f}KB({:6.1f}KB), util={:6.1f}%".format(
  #   lhs[0], lhs[1], rhs[0], rhs[1], out[0], out[1], lhs[2], rhs[2], out[2], 
  #   lhs_size / 1024, lhs_size/1024 * lhs[2], 
  #   rhs_size / 1024, rhs_size/1024 * rhs[2], 
  #   out_size / 1024, out_size/1024 * out[2], utili))
  if with_head:
    print("|size|bpe|tile (m-k0_n-k1_m-n)|ping-pong|lhs size|rhs size|out size|L1 used|L1 utili|")
    print("|----|---|--------------------|---------|--------|--------|--------|-------|-------:|")

  print("|{} KB|{}|{}x{}_{}x{}_{}x{}|{}x{}x{}|{:.1f} KB|{:.1f} KB|{:.1f} KB|{:.1f} KB|{:.2f}%|".format(
    size/1024, bpe, lhs[0], lhs[1], rhs[0], rhs[1], out[0], out[1], lhs[2], rhs[2], out[2], 
    lhs_size / 1024, rhs_size / 1024, out_size / 1024, used / 1024, utili))


if __name__ == '__main__':
  mini_batch = 4
  d_model = 512
  d_k = 64
  d_v = 64
  vocab_nums = 8192
  msl = 128
  head_nums = 8


  l1_forward_n(size=512*1024-16*1024, m=1, k=16, pp=2, bpe=2)
  l1_forward_n(size=512*1024-16*1024, m=8, k=16, pp=2, bpe=2)
  l1_forward_n(size=512*1024-16*1024, m=16, k=16, pp=2, bpe=4)
  l1_forward_n(size=512*1024-16*1024, m=16, k=16, pp=2, bpe=2)
  l1_forward_n(size=512*1024-16*1024, m=32, k=16, pp=2, bpe=2)
  l1_forward_n(size=512*1024-16*1024, m=64, k=16, pp=2, bpe=4)
  l1_forward_n(size=512*1024-16*1024, m=64, k=16, pp=2, bpe=2)
  
  # l1_forward_n(size=1024*1024-16*1024, m=1, k=16, pp=2, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=1, k=16, pp=1, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=1, k=16, pp=2, bpe=2)
  # l1_forward_n(size=1024*1024-16*1024, m=1, k=16, pp=1, bpe=2)
  # l1_forward_n(size=1024*1024-16*1024, m=32, k=16, pp=2, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=32, k=16, pp=1, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=32, k=16, pp=2, bpe=2)
  # l1_forward_n(size=1024*1024-16*1024, m=32, k=16, pp=1, bpe=2)
  # l1_forward_n(size=1024*1024-16*1024, m=64, k=16, pp=2, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=64, k=16, pp=1, bpe=4)
  # l1_forward_n(size=1024*1024-16*1024, m=64, k=16, pp=2, bpe=2)
  # l1_forward_n(size=1024*1024-16*1024, m=64, k=16, pp=1, bpe=2)

  l1_forward(size=512*1024-16*1024, lhs=[ 1, 64, 1], rhs=[4096, 16, 2], out=[ 1, 4096, 1], bpe=4, with_head=True)
  l1_forward(size=512*1024-16*1024, lhs=[ 1, 64, 1], rhs=[4096, 16, 2], out=[ 1, 4096, 1], bpe=2)
  l1_forward(size=512*1024-16*1024, lhs=[ 8, 64, 1], rhs=[4096, 16, 2], out=[ 8, 4096, 1], bpe=2)
  l1_forward(size=512*1024-16*1024, lhs=[16, 64, 1], rhs=[4096, 16, 2], out=[16, 4096, 1], bpe=2)
  l1_forward(size=512*1024-16*1024, lhs=[32, 64, 1], rhs=[4096, 16, 2], out=[32, 4096, 1], bpe=2)
  

  l1_forward(size=1024*1024-16*1024, lhs=[ 1, 64, 1], rhs=[4096, 16, 2], out=[ 1, 4096, 1], bpe=4)
  l1_forward(size=1024*1024-16*1024, lhs=[16, 64, 1], rhs=[4096, 16, 2], out=[16, 4096, 1], bpe=4)
  l1_forward(size=1024*1024-16*1024, lhs=[32, 64, 1], rhs=[4096, 16, 2], out=[32, 4096, 1], bpe=4)
  l1_forward(size=1024*1024-16*1024, lhs=[64, 64, 1], rhs=[4096, 16, 2], out=[64, 4096, 1], bpe=4)

  l1_forward(size=1024*1024-16*1024, lhs=[ 1, 64, 1], rhs=[4096, 16, 2], out=[ 1, 4096, 1], bpe=2)
  l1_forward(size=1024*1024-16*1024, lhs=[16, 64, 1], rhs=[4096, 16, 2], out=[16, 4096, 1], bpe=2)
  l1_forward(size=1024*1024-16*1024, lhs=[32, 64, 1], rhs=[4096, 16, 2], out=[32, 4096, 1], bpe=2)
  l1_forward(size=1024*1024-16*1024, lhs=[64, 64, 1], rhs=[4096, 16, 2], out=[64, 4096, 1], bpe=2)
  l1_forward(size=1024*1024-16*1024, lhs=[128, 64, 1], rhs=[4096, 16, 2], out=[128, 4096, 1], bpe=2)


  ### Test 1
  # test_embedding(None)
  # test_embedding(1.0)
  
  ### Test 2
  # pe = PositionalEncoding(d_model, 0.1, msl)
  # x = torch.zeros(mini_batch, msl, d_model)
  # y = pe.forward(x)

  ### Mult-head Attention 
  # i_q = torch.zeros(mini_batch, msl, d_model)
  # i_k = torch.zeros(mini_batch, msl, d_model)
  # i_v = torch.zeros(mini_batch, msl, d_model)
  # mult_head_att = MultiHeadedAttention(head_nums, d_model)
  # res = mult_head_att(i_q, i_k ,i_v)
  # print('mult_head_att_out.shape={}'.format(res.shape))


  # ## LayerNormal: last dim
  # dummy, batch, sentence_length, embedding_dim = 2, 3, 4, 5
  # embedding = torch.randn(dummy, batch, sentence_length, embedding_dim)
  # layer_norm = torch.nn.LayerNorm(embedding_dim)
  # # Activate module
  # activate = layer_norm(embedding)
  # print(embedding.shape)
  
  # # Image Example
  # N, C, H, W = 3, 4, 5, 6
  # input = torch.randn(N, C, H, W)
  # # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
  # # as shown in the image below
  # layer_norm = torch.nn.LayerNorm([C, H, W])
  # output = layer_norm(input)
  # print(input.shape)
  # print(output[0,0,0,:])
  
  # ln = Layer_norm()
  # activate1 = ln(input)
  # print(activate1.shape)
  # print(activate1[0,0,0,:])

