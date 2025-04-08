# DeepSeek-V3

## Script

### `convert.py`

提供`convert.py`函数可以将HF模型转换成本地脚本能够运行的模型，包括对config变量名字的转换以及进行模型并行需要的拆分。

```python
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
```

### `fp8_cast_bf16.py`

671B 模型文件是FP8 (__FP8.E4M3__) 数据类型的，对于不支持FP8的硬件可以使用本脚本将其转换成 __BF16__ 数据类型。

## Config

除了671B以外，在`config`目录下还有一个16B以及一个236B的配置文件，据说这两个模型并未能完成训练。值得注意的是DeepSeek-V2-Base是一个236B的混合专家模型。

| Parameter             | config_671B.json | config_236B.json | config_16B.json | Description                                      |
|-----------------------|------------------|------------------|-----------------|--------------------------------------------------|
| vocab_size            | 129280           | 102400           | 102400          | Vocabulary size                                  |
| dim                   | 7168             | 5120             | 2048            | Embedding dimension                              |
| inter_dim             | 18432            | 12288            | 10944           | Intermediate dimension                           |
| moe_inter_dim         | 2048             | 1536             | 1408            | Mixture of experts intermediate dimension        |
| n_layers              | 61               | 60               | 27              | Number of layers                                 |
| n_dense_layers        | 3                | 1                | 1               | Number of dense layers                           |
| n_heads               | 128              | 128              | 16              | Number of attention heads                        |
| n_routed_experts      | 256              | 160              | 64              | Number of routed experts                         |
| n_shared_experts      | 1                | 2                | 2               | Number of shared experts                         |
| n_activated_experts   | 8                | 6                | 6               | Number of activated experts                      |
| n_expert_groups       | 8                | 8                | 1               | Number of expert groups                          |
| n_limited_groups      | 4                | 3                | 1               | Number of limited groups                         |
| route_scale           | 2.5              | 16.0             | 1.0             | Routing scale factor                             |
| score_func            | sigmoid          | softmax          | softmax         | Scoring function                                 |
| q_lora_rank           | 1536             | 1536             | 0               | Q LoRA rank                                      |
| kv_lora_rank          | 512              | 512              | 512             | KV LoRA rank                                     |
| qk_nope_head_dim      | 128              | 128              | 128             | QK NOPE head dimension                           |
| qk_rope_head_dim      | 64               | 64               | 64              | QK ROPE head dimension                           |
| v_head_dim            | 128              | 128              | 128             | V head dimension                                 |
| dtype                 | fp8              | bf16             | bf16            | Data type                                        |
| mscale                | 1.0              | 1.0              | 0.707           | Scaling factor for extended attention            |
| max_batch_size        | 8                | 8                | 8               | Maximum batch size                               |
| max_seq_len           | 16384            | 16384            | 16384           | Maximum sequence length                          |
| original_seq_len      | 4096             | 4096             | 4096            | Original sequence length                         |
| rope_theta            | 10000.0          | 10000.0          | 10000.0         | Base for rotary positional encoding              |
| rope_factor           | 40               | 40               | 40              | Scaling factor for extended sequence lengths     |
| beta_fast             | 32               | 32               | 32              | Fast beta correction factor                      |
| beta_slow             | 1                | 1                | 1               | Slow beta correction factor                      |

## Architecture

