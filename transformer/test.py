from torch import LongTensor, norm
from torch.nn import Embedding

def test_max_norm(max_norm):
  sentences = LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
  embedding = Embedding(num_embeddings=10, embedding_dim=8, max_norm=max_norm)
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

if __name__ == '__main__':
  test_max_norm(None)
  test_max_norm(1.0)
