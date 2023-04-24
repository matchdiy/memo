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
    


if __name__ == '__main__':
  mini_batch = 1
  d_model = 512
  d_k = 64
  d_v = 64
  vocab_nums = 8192
  msl = 128
  head_nums = 8

  ### Test 1
  # test_embedding(None)
  # test_embedding(1.0)
  
  ### Test 2
  # pe = PositionalEncoding(d_model, 0.1, msl)
  # x = torch.zeros(mini_batch, msl, d_model)
  # y = pe.forward(x)

  i_q = torch.zeros(mini_batch, msl, d_model)
  i_k = torch.zeros(mini_batch, msl, d_model)
  i_v = torch.zeros(mini_batch, msl, d_model)
  mult_head_att = MultiHeadedAttention(head_nums, d_model)
  res = mult_head_att(i_q, i_k ,i_v)
  #print(res.shape)
