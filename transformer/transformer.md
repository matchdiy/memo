
<center><h1> Transformer </h1> </center>

<center>
<p><a href="https://arxiv.org/abs/1706.03762">Attention is All You Need
</a></p>
</center>

<h3> Table of Contents </h3>
<ul>
<li><a href="#terminology">Terminology</a></li>
<li><a href="#embedding">Embedding</a>
<ul>
<li><a href="#one-hot-encoding">One-Hot Encoding</a></li>
<li><a href="#world-embedding">World Embedding</a></li>
<li><a href="#positional-encoding">Positional Encoding</a></li>
<li><a href="#output-embedding">Output Embedding</a></li>
</ul></li>
<li><a href="#attention">Attention</a>
<ul>
<li><a href="#Scaled-Dot-Product-Attention">Scaled Dot-Product Attention</a></li>
<li><a href="#Multi-head-Attention">Multi-head Attention</a></li>
<li><a href="#Multi-head Attention-Gradient">Multi-head Attention Gradient</a></li>
</ul></li>
<li><a href="#embeddings-and-softmax">Embeddings and Softmax</a></li>
<li><a href="#positional-encoding">Positional Encoding</a></li>
<li><a href="#full-model">Full Model</a></li>
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
|head number               |$head$            | 8   |    |
|dimension of key and query|$d_k$             | 64(SD=40) | $d_v = d_{\text{model}} \div head$   |
|dimension of value        |$d_v$             | 64(SD=40) |$d_k=d_v$|
|dimension of model        |$d_{\text{model}}$| 512(SD=320) |embedding table size|
|dimension of feed forward |$d_{\text{ff}}$   | 2048 |    |
|feature                   |                  | |[ $batch$, $msl$ ]|
|query                     |Q                 | |[ $batch$, $src\_sl$, $d_{\text{model}}$ ] or [ $batch$, $msl$, $d_{\text{model}}$ ]|
|key                       |K                 | |[ $batch$, $tgt\_sl$, $d_{\text{model}}$ ] or [ $batch$, $msl$, $d_{\text{model}}$ ]|
|value                     |V                 | |[ $batch$, $tgt\_sl$, $d_{\text{model}}$ ] or [ $batch$, $msl$, $d_{\text{model}}$ ]|
|token embedding table     |                  | |[ $vocabs$, $d_{\text{model}}$ ]|
|position embedding table  |                  | |[ $msl$, $d_{\text{model}}$ ]|
|weight of query           |QW                | |[ $d_{\text{model}}$, $d_{\text{model}}$ ]|
|weight of key             |KW                | |[ $d_{\text{model}}$, $d_{\text{model}}$ ]|
|weight of value           |VW                | |[ $d_{\text{model}}$, $d_{\text{model}}$ ]|
|weight of output          |WO                | |[ $d_{\text{model}}$, $d_{\text{model}}$ ]|
|weight 1 of layernorm     |l1                | |[ $d_{\text{model}}$ ]|
|weight 1 of FFN           |W1                | |[ $d_{\text{model}}$, $d_{\text{ff}}$ ]|
|bias 1 of FFN             |b1                | |[ $d_{\text{model}}$ ]|
|weight 2 of layernorm     |l2                | |[ $d_{\text{model}}$ ]|
|weight 2 of FFN           |W2                | |[ $d_{\text{ff}}$, $d_{\text{model}}$ ]|
|bias 2 of FFN             |b2                | |[ $d_{\text{model}}$ ]|

* $msl$ 和 $vocabs$ 的值取决于数据集
* $d_{\text{ff}}$: 一般习惯 $d_{\text{ff}}=4 \cdot d_{\text{model}}$，但论文没有说明这一点
* QW，KW，VW: 无论 $head$ 是多少这几个权重的shape都是 [ $d_{\text{model}}$, $d_{\text{model}}$ ]，多头是指多个attention以sub的方式共同使用W，于是才有：$d_v=d_{\text{model}}/head$

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

Attention的三个输入 Q(qurey), K(Key), V(value) 拥有相同的Shap，他们是这样计算的：

$$
Q[ batch, msl, d_q ] = Input[ batch, msl, d_{\text{model}} ] * QW[ d_{\text{model}}, d_q ]
$$

$$
K[ batch, msl, d_k ] = Input[ batch, msl, d_{\text{model}} ] * KW[ d_{\text{model}}, d_k ]
$$

$$
V[ batch, msl, d_v ] = Input[ batch, msl, d_{\text{model}} ] * VW[ d_{\text{model}}, d_v ]
$$

__注意__: $d_q, d_q, d_v$ 一般是相同的，但是在group head的时候也可以不同。另外一般 $head=1$ 时 $d_k==d_{\text{model}}$

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

## Multi-head Attention

多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息.

$$
\mathrm{MultiHead}(Q, K, V) =
    \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

```Python

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
```

## MHA 计算过程以及精度

* __(1) 计算Q，K，V三个变量，他们是 MultHeadAtten 的输入，不需要融合进来__

$$
Q[ batch, msl, d_{\text{model}} ] = Dot(Input[ batch, msl, d_{\text{model}} ], QW[ d_{\text{model}}, d_{\text{model}} ])
$$

$$
K[ batch, msl, d_{\text{model}} ] = Dot(Input[ batch, msl, d_{\text{model}} ], WK[ d_{\text{model}}, d_{\text{model}} ])
$$

$$
V[ batch, msl, d_{\text{model}} ] = Dot(Input[ batch, msl, d_{\text{model}} ], WV[ d_{\text{model}}, d_{\text{model}} ])
$$

* __(2) 拆出 $head$__

$$
Q[ batch, msl, head, d_k ] = Reshape(Q[ batch, msl, d_{\text{model}} ])
$$

$$
K[ batch, msl, head, d_k ] = Reshape(K[ batch, msl, d_{\text{model}} ])
$$

$$
V[ batch, msl, head, d_v ] = Reshape(V[ batch, msl, d_{\text{model}} ])
$$

* __(3) 计算 $QK^T$__
  * 计算式
  $$
  QK^T[ batch, msl_m, head, msl_n ] = Dot(Q[ batch, msl_m, head, d_k ], K[ batch, msl_n, head, d_k ], lhs\_batch\_dims=\{0,2\}, rhs\_batch\_dims=\{0,2\}, lhs\_contracting\_dims=\{3\}, rhs\_contracting\_dims=\{3\}, out\_batch\_dims=\{0,2\})
  $$
  * _FIMXE：AMP时输入和输出都是 FP16_
  * 注意：$msl_m==msl_n$ 这里只是为了标注出不同的维度意义。

* __(4) 计算 Scale__
  * $QK^T[ batch, msl, head, msl ] = Mul(QK^T[ batch, msl, head, msl ], 1/\sqrt{d_{\text{model}}})$
  * _FIXME：AMP时使用 FP16 输入输出_
  
* __(5) 选项 MaskFill__
  * $QK^T[ batch, msl, head, msl ] = MaskFill(QK^T[ batch, msl, head, msl ] == 0, -1e9)$
  * mask value是负无穷，这样的目的是让mask==0的地方经过softmax后仍然是0.
  * _FIXME：AMP时使用 FP16 输入和输出_

* __(6) 计算 Softmax__
  * $QK^T[ batch, msl, head, msl ] = Softmax(QK^T[ batch, msl, head, msl ], dim=-1)$
  * _FIXME：AMP时，如果输入是FP16，那么计算过程中需要先转成FP32再计算，输出转成 FP16_

* __(7) 选项 Dropout__
  * $QK^T[ batch, msl, head, msl ] = Dropout(QK^T[ batch, msl, head, msl ], p = 0.1)$
  * _FIXME：如果是基于当前Tiling部分做Dropout，这里恐怕是有算法上的风险的_

* __(8) 计算Out__
  * 计算式：
  $$
      Attn[ batch, msl, head, d_v ] = Dot(PAttn[ batch, msl, head, msl ], V[ batch, msl, head, d_v ], lhs\_batch\_dims=\{0,2\}, rhs\_batch\_dims=\{0,2\}, lhs\_contracting\_dims=\{3\}, rhs\_contracting\_dims=\{1\}, out\_batch\_dims=\{0,2\})
  $$
  * _FIXME: AMP的时候输入和输出使用FP16；MHA Fusion到此为止也许就可以了，继续fuse的话反向还是需要计算出这个结果_。

* __(x) 重新合并 $head$__
  * $Attn[ batch, msl, d_{\text{model}} ] = Reshape(Attn[ batch, msl, head, d_v ])$
* __(x) 最后一个Linear__
  * $Out[ batch, msl, d_{\text{model}} ] = Dot(Attn[ batch, msl, d_{\text{model}} ], WO[ d_{\text{model}}, d_{\text{model}} ])$
  * _FIXME: 这一步的计算不需要Fusion到MHA中，否则前面的Dot在反向计算的时候需要重新计算出来，这里需要进行权衡。_

这样的计算流程可以去掉所有的 transpose 操作，网络中后续的计算都是按照 $Out[batch, msl, d_{\text{model}}]$ layout进行的。

## MHA 计算负载

|$Args/Models$|SD Clip|SD Unet|Bert Base|Bert Large|GPT-2 XL|GPT-3|GPT-4 (Est.)|
|-------------|-------|-------|---------|----------|--------|-----|------------|
|$msl$             |77|32400|512|512|2048|3072|32000|
|$d_{\text{model}}$|768|320|768|1024|1600|12288|25600|
|$head$            |12|5|12|16|25|96|160|

## MHA 算子实现

根据上面描述的计算过程和计算负载可以发现，$QK^T[batch, msl_m, head, msl_n]$ 在一些计算任务中将会是一个比较大的 Teansor，系统在这里会遇到存储瓶颈。我们需要实现的MHA算子需要将这个巨大的Tensor分片后隐藏到 L1、L2，或者只占用较少的L3的条件下，完成这个计算。训练过程中反向计算的时候需要重新计算出这个 Tensor，会有额外的计算量。由此可见这个MHA算子本身并不会有直接的性能提升，而是用计算换取存储的优化，保证达模型功能。优化存储有助于提高BatchSize，使得系统能够有更高的利用率。

### MHA 算子实现：Cluster Level 并行

从计算过程中可以确定 $batch$ 和 $msl_m$ 这两个维度是可以向后传递的，4C Split 作用在这两个维度上的话不会导致反复发生Split-Merge，以便于提高整体性能。选择 $batch$ 维度可能的风险是他的数值在大模型训练任务中可能会比较小，无法让整个SOC满负载工作。而选择 $msl_m$ 的话不会发生这种负载不够或者不均衡的情况，他的数值相对于ClusterNums而言已经足够大了，但是切分 $msl_m$ 维度会导致 RHS 需要完整进入到每一个Cluster，对整个SOC而言，全部Cluster上的RHS是重复的。从性能优化的角度应该优先切分 $batch$ ，但应该考虑利用率：

  $$
  utilization = { double(batch) \over ((batch + cluster_{\text{muns}} - 1) / cluster_{\text{muns}}) * cluster_{\text{muns}}}
  $$

* 如果 utilization 大于阈值（比如 80%）应该优先选择将 $batch$ 切分到不同的Cluster上并行
* 如果 utilization 小于阈值，那么应该优先将 $msl_m$切分到不同的Cluster上并行。
* 阈值需要根据 RHS 数据量的大小进行计算，到底是选择负载均衡还是选择数据重复搬运，需要权衡。

接下来我们只讨论一个 Cluster 内部的计算。

### MHA 算子实现：Layout

我们使用下面的符号来描述分配到一个Cluster上的计算任务。$batch$ 和 $ms_m$ 都可能被切分，这里开始使用 $b0$ 和 $m$来表示；为了更清晰表达BatchDot的计算，使用$b1$=$head$，表示这也是一个Batch维度。这里需要注意的是XLA算子定义中支持 BatchDot的输入支持 `lhs_batch_dims={}` 和`rhs_batch_dims={}`都是多维并且可以是不连续维度的，但是输出要求一定是 ___OutputLayout={Batch, M, N}___ ，XLA的这种定义导致了Transformer中需要多次的 Transpose 操作，因为 $head$ 和 $d_k$ 在这里虽然是被切分开成为了两个维度，但是后面的计算中他们还会合成 $d_{\text{model}} = head*d_k$，所以这里需要实现的是输出 ___OutputLayout___={$batch$, $msl_m$, $head$, $msl_n$} 和输入一样，保持两个分开的Batch维度。

$msl_n$ 维度是后续Softmax计算中需要的完整维度，这个维度的数据如果能够完整保存到L1的话，那么后续的计算时可以在L1上完成传递的；如果不行那么就需要完整保存到L2上，这样可能会引起SIP读写L2发生`bank conflict`；如果还是无法完整放在L2的话，那么需要在L3上开一个临时buffer，接下来分别进行分析。假设 $QK^T$计算优先在 $batch$, $msl_m$ 以及 $head$ 维度进行了切分，切分后分别是 $b0$, $m$以及$b1$，那么计算可以表示成：

  $$
    QK^T[b0, m, b1, msl_n] = Dot(Q[b0, m, b1, d_k], K[b0, msl_n, b1, d_k], lhs\_batch\_dims=\{0,2\}, rhs\_batch\_dims=\{0,2\}, lhs\_contracting\_dims=\{3\}, rhs\_contracting\_dims=\{3\}, out\_batch\_dims=\{0,2\})
  $$

### MHA 算子实现：通过L1交换数据

这个方案要求每个Sip都能计算出完整的$msl_n$，这就要求$msl_n$的值不能过大。而我们实现MHA融合算子的目的是为了解决$msl_n$过大给系统带来的存储压力，所以这种条件下合理的支持范围是比较小的。在Dorado上拥有1M的L1，相对比Pavo的512KB L1 能够支持的范围要大的多。

#### L1::QKt (Dot)

ConvGen有计算限制，如果不满足这些最小size的限制，需要用户padiding到最小sizem，比如 $min(k)=16$，在这个约束条件下，我们计算一下 $n=(1024*1024-m*k*bpe*pp)/(m*bpe + k*bpe*pp)$其中 pp=ping-pong。L1 Size在pavo和dorado上分别是512KB和1MB，但要预留16K stack。这里使用 $d_k=64$进行的计算，实际上SD模型的$d_k=40$，需要padding到48

|size|bpe|tile (m-k0_n-k1_m-n)|ping-pong|lhs size|rhs size|out size|L1 used|L1 utili|
|----|---|--------------------|---------|--------|--------|--------|-------|-------:|
|496.0 KB|4|1x64_4096x16_1x4096|1x2x1|0.2 KB|256.0 KB|16.0 KB|528.2 KB|106.50%|
|496.0 KB|2|1x64_4096x16_1x4096|1x2x1|0.1 KB|128.0 KB|8.0 KB|264.1 KB|53.25%|
|496.0 KB|2|8x64_4096x16_8x4096|1x2x1|1.0 KB|128.0 KB|64.0 KB|321.0 KB|64.72%|
|496.0 KB|2|__16x64_4096x16_16x4096__|1x2x1|2.0 KB|128.0 KB|128.0 KB|386.0 KB|__77.82%__|
|496.0 KB|2|32x64_4096x16_32x4096|1x2x1|4.0 KB|128.0 KB|256.0 KB|516.0 KB|104.03%|
|1008.0 KB|4|1x64_4096x16_1x4096|1x2x1|0.2 KB|256.0 KB|16.0 KB|528.2 KB|52.41%|
|1008.0 KB|4|16x64_4096x16_16x4096|1x2x1|4.0 KB|256.0 KB|256.0 KB|772.0 KB|76.59%|
|1008.0 KB|4|32x64_4096x16_32x4096|1x2x1|8.0 KB|256.0 KB|512.0 KB|1032.0 KB|102.38%|
|1008.0 KB|4|64x64_4096x16_64x4096|1x2x1|16.0 KB|256.0 KB|1024.0 KB|1552.0 KB|153.97%|
|1008.0 KB|2|1x64_4096x16_1x4096|1x2x1|0.1 KB|128.0 KB|8.0 KB|264.1 KB|26.20%|
|1008.0 KB|2|16x64_4096x16_16x4096|1x2x1|2.0 KB|128.0 KB|128.0 KB|386.0 KB|38.29%|
|1008.0 KB|2|32x64_4096x16_32x4096|1x2x1|4.0 KB|128.0 KB|256.0 KB|516.0 KB|51.19%|
|1008.0 KB|2|__64x64_4096x16_64x4096__|1x2x1|8.0 KB|128.0 KB|512.0 KB|776.0 KB|__76.98%__|
|1008.0 KB|2|128x64_4096x16_128x4096|1x2x1|16.0 KB|128.0 KB|1024.0 KB|1296.0 KB|128.57%|

通过L1交换数据的方案，Pavo上无法支持BPE=4，只能用混精来处理（如果输入数据是FP32或者是EF32的，那么需要先将其convert成FP16）；Dorado上是可以支持BPE=4，当使用混精时可以支持 $m=64$，刚好可以存满一个VR。由于第一个算子是个Dot计算，所以我们可以假设任何情况下将其转成 FP16/BF16 进行处理都可以达到精度要求。

这样可以得到一些结论：使用混精实现这个算子的前提下Pavo和Dorado都可以基于L1方案支持 $msl=4096$，输入输出使用FP16/BF16，FP32/EF32输入时在本算子内部进行convert。(__大模型中BF16也很常见，我们统一用f16代表FP16和BF16__)。

* L1 Tiling
  |platform|lhs  |rhs      |out    |选择倾向|
  |-|-----|---------|-------|-|
  |Pavo  |16x64|4096x16x2(256KB)|16x4096(128KB)|需要convgen输出layout{n,m}|
  |Dorado|64x64|4096x16x2(256KB)|64x4096(512KB)|需要convgen输出layout{n,m}|
  * 必须让b1=1，这样可以避免transpose操作，如果有空间剩余可以通过增加b0来调节。
  * lhs k 应该尽量大，最好是完整的dv，以便于减少DMA配置次数。
  * lhs和out不需要ping-pong，rhs需要ping-pong buffer
  * out 在L1上从address=0x00开始分配，lhs和rhs依次在其后面分配，这样可以减少碎片。
  * L1上要预留16KB stack，注意无法使用完整的L1
  * $out[ 4096, 64 ] = ConvGenDot(rhs[ 4096, 16 ], lhs[ 64, 16] )$，交换lhs和rhs能够让ConvGen输出成我们想要的layout{n,m}，这样可以消除反复的transpose操作，__下图描述了需要反复transpose的原因__: ![Tux, the Linux mascot](/transformer/scale-maskfill-reducemax.png)

我们把全部计算流程走完，再讨论 L2 tiling。

#### L1::Scale-MaskFill-Max

Softmax计算中需要的 Max 计算可以在前面计算结果保存在VR中的时期内完成，提前到这里和 Mul 一起完成。如果选项MaskedFill是需要的，那么要在mask-fill后计算最大值。

* 一下计算在VR中完成：
  * $m=64$ 是 $m=16$ 的特例，我们基于$m=16$描述计算过程。
  * $inout[ 4096，16 ], row_max[16] = ScaleMaskFillReduceMaxKernel(inout[ 4096, 16 ])$, 数据类型为FP16/BF16
  * (1) 计算 mul，结果保存在VR中
  * (2) 如果使能了MaskedFill，需要在L1上开一个临时buffer，比如：$mask[1024, 16]$ 计算过程中需要用sdma反复Slice数据到L1，计算结果在VR中，然后一边Store到L1，一边基于VR计算ReduceMax，需要将 vload inout, vload mask, vmul, compare, vstore 这些操作进行流水优化。
  * (3) inout=ReduceMax(inout)，写回L1上原来的位置。

#### L1::Softmax

为了保证精度这里超越函数计算需要使用f32，这样带来的问题就是计算的中间结果无法写回L1 的 inoput buffer。我们初步打算利用VACC（128KB）寄存器来完成中间结果的保存。

* 以下计算在VR/VA中完成：
  * [VR] $x_{ij}=x_{ij}-max_i$, datatype=f16
  * [VR] $x_{ij} = convert(x_{ij})$, datatype=f16 to f32
  * [VR] $e^{x_{ij}}=exp(x_{ij})$, datatype=f32
  * [VA] $e^{x_{ij}}=move(e^{x_{ij}})$, datatype=f32
    * _MOVVR2VA_，把VR中计算出来的exp保存到VA中，直到可以计算sum为止。同时保留已经计算出来的结果。
    * 编译器可能指允许使用一半，超过了会发生spilling。用汇编吗？可能也没有多少代码，用汇编也许是更好的。
  * [VA] $sum_i=sum(e^{x_{ij}, dim=i})$
    * _MOP.MADD VACC0,VACC0,VACC4_，
  * [VR] $result(i,j)=e^{x_{ij}} / sum(i)$ datatype=f32
    * 将 $e^{x_{ij}}$ 和 对应的 $sum_i$ 从VA中取出到VR，然后计算最终结果。
  * [VR] convert result form fp32 to fp16
  * [VR] store to L1 (inout)

#### L1::Linear

* 计算公式：
  $$
    QK^T[batch, msl_m, head, d_v] = Dot(Attn[batch, msl_m, head, msl_n], V[batch, msl_n, head, d_v], lhs\_batch\_dims=\{0,2\}, rhs\_batch\_dims=\{0,2\}, lhs\_contracting\_dims=\{3\}, rhs\_contracting\_dims=\{1\}, out\_batch\_dims=\{0,2\})
  $$

* L1 Tiling

|size|bpe|tile (m-k0_n-k1_m-n)|ping-pong|lhs size|rhs size|out size|L1 used|L1 utili|
|----|---|--------------------|---------|--------|--------|--------|-------|-------:|
|496.0 KB|2|16x4096_4096x40_16x40|1x1x1|128.0 KB|320.0 KB|1.2 KB|449.2 KB|90.57%|
|496.0 KB|2|__16x4096_2048x40_16x40__|1x2x1|128.0 KB|160.0 KB|1.2 KB|449.2 KB|__90.57%__|
|496.0 KB|2|16x4096_1024x40_16x40|1x2x1|128.0 KB|80.0 KB|1.2 KB|289.2 KB|58.32%|
|1008.0 KB|2|64x4096_4096x40_64x40|1x1x1|512.0 KB|320.0 KB|5.0 KB|837.0 KB|83.04%|
|1008.0 KB|2|__64x4096_2048x40_64x40__|1x2x1|512.0 KB|160.0 KB|5.0 KB|837.0 KB|__83.04%__|
|1008.0 KB|2|64x4096_1024x40_64x40|1x2x1|512.0 KB|80.0 KB|5.0 KB|677.0 KB|67.16%|

* 计算
  * $out[ b0, m, 1, d_v] = Dot(attn[ b0, m, 1, msl_n ], v[ b0, msl_n, 1, d_v ])$
  * 在SD模型中 d_v=40， 那么这里的计算无论是否切reduce维度都是可以放下的，但是如果d_v=64的话就必须要切rhs的reduce维度。为了保证通用性，以及支持L2上更灵活的切分方式，希望这里仍旧按照切rhs的reduce维度进行实现，这种条件下是需要开ping-pong的。
  * out 写回L2，至此完成了L1上的一次完整MHA，接下来就是继续从L2读取后续的切片反复这个过程。

#### L2::QKt (Dot)

|level|size|bpe|tile (m-k0_n-k1_m-n)|ping-pong|lhs size|rhs size|out size|L1 used|L1 utili|
|-----|----|---|--------------------|---------|--------|--------|--------|-------|-------:|
| L1  |496.0 KB|2|__16x64_4096x16_16x4096__|1x2x1|2.0 KB|128.0 KB|128.0 KB|386.0 KB|__77.82%__|
| L1  |1008.0 KB|2|__64x64_4096x16_64x4096__|1x2x1|8.0 KB|128.0 KB|512.0 KB|776.0 KB|__76.98%__|

### MHA 算子实现：通过L2交换数据

### MHA 算子实现：通过L3交换数据

## Multi-head Attention Gradient

TODO
