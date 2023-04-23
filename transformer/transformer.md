
<center><h1> Transformer </h1> </center>

<center>
<p><a href="https://arxiv.org/abs/1706.03762">Attention is All You Need
</a></p>
</center>

<h3> Table of Contents </h3>
<ul>
<li><a href="#terminology">Terminology</a></li>
<li><a href="#embedding">Embedding</a><ul>
<li><a href="#one-hot-encoding">One-Hot Encoding</a></li>
<li><a href="#world-embedding">World Embedding</a></li>
<li><a href="#positional-encoding">Positional Encoding</a></li>
<li><a href="#output-embedding">Output Embedding</a></li>

</ul></li>
<li><a href="#attention">Attention</a></li>
<li><a href="#part-1-model-architecture">Part 1: Model Architecture</a></li>
<li><a href="#model-architecture">Model Architecture</a><ul>
<li><a href="#encoder-and-decoder-stacks">Encoder and Decoder Stacks</a></li>
<li><a href="#position-wise-feed-forward-networks">Position-wise Feed-Forward
Networks</a></li>
<li><a href="#embeddings-and-softmax">Embeddings and Softmax</a></li>
<li><a href="#positional-encoding">Positional Encoding</a></li>
<li><a href="#full-model">Full Model</a></li>
<li><a href="#inference">Inference:</a></li>
</ul></li>
<li><a href="#part-2-model-training">Part 2: Model Training</a></li>
</ul>

# Terminology

|Term                      |Abbr              |Value|Shape|
|----                      |----              |-----|-----------|
|layer                     |$N$               | 6   |           |
|batch size                |$batch$           |     |           |
|max sequence length       |$msl$             |128  |dataset base |
|src sequence length       |$src\_sl$         |any  |sequence base|
|target sequence length    |$tgt\_msl$        |any  |sequence base|
|$batch * msl$             |$M$               |     |             |
|vocab size                |$vocabs$          |50257|dataset base|
|head number               |$head$               | 8   |    |
|dimension of key and query|$d_k$             | 64  |    |
|dimension of value        |$d_v$             | 64  |$d_k=d_v$|
|dimension of model        |$d_{\text{model}}$| 512 |$d_{\text{model}}=h \cdot d_v$|
|dimension of feed forward |$d_{\text{ff}}$   | 2048|    |
|feature                   |                  | |[$batch$, $msl$]|
|query                     |Q                 | |[$batch$, $src\_sl$, $d_{\text{model}}$] or [$batch$, $msl$, $d_{\text{model}}$]|
|key                       |K                 | |[$batch$, $tgt\_sl$, $d_{\text{model}}$] or [$batch$, $msl$, $d_{\text{model}}$]|
|value                     |V                 | |[$batch$, $tgt\_sl$, $d_{\text{model}}$] or [$batch$, $msl$, $d_{\text{model}}$]|
|token embedding table     |                  | |[$vocabs$, $d_{\text{model}}$]|
|position embedding table  |                  | |[$msl$, $d_{\text{model}}$]|
|weight of query           |WQ                | |[$head$, $d_{\text{model}}$, $d_k$]|
|weight of key             |WK                | |[$head$, $d_{\text{model}}$, $d_k$]|
|weight of value           |WV                | |[$head$, $d_{\text{model}}$, $d_v$]|
|weight of output          |WO                | |[$h \cdot d_v$, $d_{\text{model}}$]|
|weight 1 of layernorm     |l1                | |[$d_{\text{model}}$]|
|weight 1 of FFN           |W1                | |[$d_{\text{model}}$, $d_{\text{ff}}$]|
|bias 1 of FFN             |b1                | |[$d_{\text{model}}$]|
|weight 2 of layernorm     |l2                | |[$d_{\text{model}}$]|
|weight 2 of FFN           |W2                | |[$d_{\text{ff}}$, $d_{\text{model}}$]|
|bias 2 of FFN             |b2                | |[$d_{\text{model}}$]|

* $msl$ 和 $vocabs$ 的值取决于数据集
* $d_{\text{ff}}$: 一般习惯 $d_{\text{ff}}=4 \cdot d_{\text{model}}$，但论文没有说明这一点
* WO: 一般习惯 $d_{\text{model}}=h \cdot d_v$, 但这不是必须的。multi-head 输出时需要concat多个 $head$ 个 $d_v$

# Embedding

在`sequence transduction models`中，我们需要对输入序列和输出序列所使用的字典进行编码。

* 对话
  * input="how are you"
  * output="i am fine"
* 翻译
  * input="how are you"
  * output="你好吗"

## One-Hot Encoding

简单的给每个词进行编码，编码长度等于字典中词的个数，比如一个字典有5个词：

|index|vocabulary|     encode    |
|-----|----------|---------------|
| 1   | how      | [1 0 0 0 0 0] |
| 2   | are      | [0 1 0 0 0 0] |
| 3   | you      | [0 0 1 0 0 0] |
| 4   | i        | [0 0 0 1 0 0] |
| 5   | am       | [0 0 0 0 1 0] |
| 6   | fine     | [0 0 0 0 0 1] |

这种编码方法比较浪费数据，每个词的编码长度都是字典的大小，并且过于简单，无法表达词与词之间的关系。

## World Embedding

使用一个可学习的权重矩阵 __W__[$vocabs$, $d_{\text{model}}$] 当成一个查找表，因为 $vocabs$ 远远大于 $d_{\text{model}}$， 这样就能够把一个 OneHot[$vocabs$, $vocabs$] 编码表进行压缩, 比如 $d_{\text{model}}=3$：

```Python
W = [[w01, w02, w03],
     [w11, w12, w13],
     [w21, w22, w23],
     [w31, w32, w33],
     [w41, w42, w43],
     [w51, w52, w53]]
```

训练中可以对这个权重矩阵不断更新，让其能够学习到词与词之间的关系，比如 "i" 后面很大概率是 "am"。一般而言是可以让`Input Embedding` 和 `Output Embedding` 共享同一个权重矩阵 W。对于 `Output Embedding` 通常使用学习到的 W 进行线性变换，以及使用Softmax对下一个要出现的词进行预测。

AnnotatedTransformer中的代码如下：

```Python
import torch
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = torch.nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

torch.nn.Embedding 接口参考：[torch/nn/modules/sparse.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/sparse.py#L13)
torch.nn.Embedding 实现参考：[torch/csrc/api/src/nn/modules/embedding.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/embedding.cpp#L16)

```Python
  # torch
  class Embedding(Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None) -> None:

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

```

* 结合代码解析一下这些参数的意义：
  * num_embeddings: 字典中词总量
  * embedding_dim：$d_{\text{model}}$，world embedding 中每个词的编码长度
  * pre-trained 的时候需要设置 weight 并 freeze=True，这种情况下实际就是一个 lookup table。
  * transformer 选择对 W 进行随机初始化(当然也可以选择 pre-trained 的结果)，并且设 freeze=False，这样会在训练过程中更新 W。
  * padding_idx: 指定 W 中的一行，将其值全部设为 0，这相当于在 W 中设定了一行不会产生任何影响的编码，也就是定义了一个没有意义的词(None)，在训练过程中这一行也不会被更新(更新后也还是 0). 一个mini batch 中的每一条语句是无法保证相同长度的，那么需要用 padding_idx 对应的编码补齐，使得一个 mini batch中的语句都是相同长度，这样便于并行计算。
  * max_norm: 设定 `embedding vector` 的最大范数，如果选定行的范数大于 max_norm，会重新将这一行用max_norm正则，并且更新 W 中的对应行。
  * scale_grad_by_freq: 根据单词在一个 mini batch 中出现的频率来调整梯度Scale。

## Positional Encoding

world embedding 解决了两个问题，一是词与词之间的关联，二是压缩了字典的编码，但是只依靠world embedding并不能包含词在句子中的相对位置信息。RNN 是按顺序对句子进行处理的，一次一个 word，所以在任何地方都可以对同一个 word 使用同样的embedding vector。但是在 Transformer 中，输入句子的所有 word 是同时处理的，这个模型在可以有更高的并行性，但结构上是没有考虑词的排序和位置信息。对此，Transformer 的作者提出了加入 positional encoding 的方法来解决这个问题。

论文中给出的计算公式：

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

* 其中 $pos$ 是词在句子中的位置，句子的最大长度 $max\_sequnce\_size$，那么 $PE[ max\_sequnce\_size, d_{\text{model}} ]$
* 其中 $2i$ 和 $2i+1$ 表示的是 $[0, d_{\text{model}})$中的偶数和奇数，注意在cos计算中仍旧使用的是 $2i$，这样来让 $d_{\text{model}}$ 的不同维度上的PE有不同的值。
  * ${(pos,2i)}$ 表示一条语句中的第 $pos$ 个词的embeding vector中的第 $2i$ 个元素
  * ${(pos,2i+1)}$ 表示一条语句中的第 $pos$ 个词的embeding vector中的第 $2i+1$ 个元素
* $PE$公式保证了相同的词在不同的位置的编码一定不同 ？？？
* $PE$会和 world embedding 相加并且对其结果进行 dropout 后输出，对于Encoder和Decoder都是一样。这里为什么用 dropout 进行一次正则？？？

```Python
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
```

## Output Embedding

* 术语：
  * `Decoder Output Embedding` 是标签数据，用于和模型的输出（预测值）进行比较，是 Decoder 的输入。
  * `Decoder Output` 是模型的输出（预测值）。
* 论文中在`Decoder Output Embedding` 的输入上有 `outputs_sequnce(shifted right)`的标注，这里为什么要右移？
  * `Encoder Input Embeding`："How are you`<EOS>`"
  * `Decoder Output Embedding`: "`<BOS>` I am fine `<EOS>`"。需要的只是 "I am fine `<EOS>`"，所以要标记 shifted right

# Attention

## Scaled Dot-Product Attention

Attention的三个输入 Q(qurey), K(Key), V(value) 拥有相同的Shape：

* $ Q[batch, msl, d_k] = Input[batch, msl, d_{\text{model}}] * WQ[d_{\text{model}}, d_k]$
* $ K[batch, msl, d_k] = Input[batch, msl, d_{\text{model}}] * WK[d_{\text{model}}, d_k]$
* $ V[batch, msl, d_v] = Input[batch, msl, d_{\text{model}}] * WV[d_{\text{model}}, d_v]$

$$
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

```Python
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
```

## Multi-head attention

多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息.

$$
\mathrm{MultiHead}(Q, K, V) =
    \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

* $ Q[batch, head, msl, d_k] = Input[batch, msl, d_{\text{model}}] * WQ[head, d_{\text{model}}, d_k]$
* $ K[batch, head, msl, d_k] = Input[batch, msl, d_{\text{model}}] * WK[head, d_{\text{model}}, d_k]$
* $ V[batch, head, msl, d_v] = Input[batch, msl, d_{\text{model}}] * WV[head, d_{\text{model}}, d_v]$

----

* $tensor_1[batch_{\text{vocab}}, batch_{\text{vocab}}] = Dot(Q[batch_{\text{vocab}}, d_k], K[batch_{\text{vocab}}, d_k]^T)$
* $tensor_2[batch_{\text{vocab}}, batch_{\text{vocab}}] = Mul(tensor_1[batch_{\text{vocab}}, batch_{\text{vocab}}], 1/\sqrt{d_{\text{model}}})$
* $tensor_3[batch_{\text{vocab}}, batch_{\text{vocab}}] = softmax(tensor_2[batch_{\text{vocab}}, batch_{\text{vocab}}])$
* $tensor_4[batch_{\text{vocab}}, d_v] = Dot(tensor_4[batch_{\text{vocab}}, batch_{\text{vocab}}], V[batch_{\text{vocab}}, d_v])$
