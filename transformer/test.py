import torch
from torch import LongTensor, norm
from torch.nn import Embedding
import math
import numpy as np
import pandas as pd
import altair as alt


d_model = 4
vocab_nums = 10
msl = 32
mini_batch = 1

def test_embedding(max_norm):
  sentences = LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
  embedding = Embedding(num_embeddings=vocab_nums, embedding_dim=d_model, max_norm=max_norm)
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

    def __init__(self, d_model, dropout, max_len=msl):
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

if __name__ == '__main__':
  ### Test 1
  # test_embedding(None)
  # test_embedding(1.0)
  ### Test 2
  pe = PositionalEncoding(d_model, 0.1, msl)
  x = torch.zeros(mini_batch, msl, d_model)
  y = pe.forward(x)
