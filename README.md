<!-- markdownlint-disable MD001 MD041 -->
## INT8 KVCache

vLLM使用多个`attention_backend` 定义在 `class AttentionBackendEnum()` 包含 `FLASH_ATTN` `FLASHINFER` 等 支持自定义backend，所以我们在这里加一个`INT8_PAGE_ATTN = "vllm.v1.attention.backends.pagedint8.Int8PageAttentionBackend"`

`backends.pagedint8` 就是我们的新backend，大致是从原来的FLASH_ATTN扒下来的 把原来的算子调用注释掉写自己的pytorch版本实现 

主要的改动在`class FlashAttentionImpl(AttentionImpl):`

在init阶段添加动态量化的scale因子 需要储存因为后面需要dequantize

主要量化过程体现在`forward`函数和`do_kv_cache_update`里

### **Forward 函数**

```python
# 把初始化好的per head 量化 scale [Hkv] 转换成接口要求的[batch, Hkv]
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads) # [batch, Hkv]
            k_descale = self._int8_k_scale.view(1, -1).expand(descale_shape).contiguous()
            v_descale = self._int8_v_scale.view(1, -1).expand(descale_shape).contiguous()
```

```python
# 调新的pytorch算子
# 使用新的INT8 PagedAttention
from vllm.v1.attention.backends.pagedint8_util import paged_attention_pytorch
paged_attention_pytorch(
    q=query_actual, # 这个类型应该是torch.bfloat16
    key_cache=key_cache,
    value_cache=value_cache,
    cu_seqlens_q=attn_metadata.query_start_loc,
    block_table=attn_metadata.block_table, # 块表
    seqused_k=seqused_k, # 【batch】 每个序列的有效k长度（不包含padding）
    softmax_scale=self.scale, # sqrt(d)
    causal=attn_metadata.causal, 
    k_descale=k_descale, # k反量化系数 [batch, Hkv]
    v_descale=v_descale,    # v反量化系数 [batch, Hkv]
    out=output_actual,  
    window_size=sliding_window_size,
    alibi_slopes=self.alibi_slopes,
    softcap=self.logits_soft_cap,
)
```

### **do_kv_cache_update**

```python
  # ---------------- KV cache update ----------------
  # 这里是更新kv cache的函数，注意如果是int8 kv cache，在这里进行量化并写入cache；如果是非int8 kv cache，则不需要更新（因为上游已经更新好了）
def do_kv_cache_update(
    self,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return

    key_cache, value_cache = kv_cache.unbind(0)

    if self.is_int8_kv:
        # 1) init scales once (stop-gap)
        if self._int8_k_scale is None or self._int8_v_scale is None:
            # key.shape通常是 [T, Hkv, D]（T=token数，Hkv=KV头数，D=头维度）
            # dim=(0,2)：沿token维度(0)和头维度(D)取最大值，保留KV头维度(1)
            k_abs_max = key.abs().amax(dim=(0, 2), keepdim=False)  # [Hkv]
            v_abs_max = value.abs().amax(dim=(0, 2), keepdim=False)

            # 2) quantize: round + clamp
            ## 量化
            # x_int8 = round(x_fp32 / scale)  
            self._int8_k_scale = (k_abs_max / 127.0).clamp_min(1e-6).to(key.device)
            self._int8_v_scale = (v_abs_max / 127.0).clamp_min(1e-6).to(key.device)

        k_scale = self._int8_k_scale
        v_scale = self._int8_v_scale

        # 反量化
        # result_fp32 = result_int32 * (scale_x * scale_y)
        ks = k_scale.view(1, -1, 1)
        vs = v_scale.view(1, -1, 1)

        key_int8 = torch.round((key / ks).to(torch.float32)).clamp(-127, 127).to(torch.int8)
        value_int8 = torch.round((value / vs).to(torch.float32)).clamp(-127, 127).to(torch.int8)

        # 3) write into paged cache using slot_mapping
        key_cache_i8 = key_cache.view(torch.int8)
        value_cache_i8 = value_cache.view(torch.int8)

        # 调用我们pytorch的实现
        from vllm.v1.attention.backends.pagedint8_util import reshape_and_cache_flash_pytorch
        reshape_and_cache_flash_pytorch(
            key_int8=key_int8,
            value_int8=value_int8,
            key_cache_i8=key_cache_i8,
            value_cache_i8=value_cache_i8,
            slot_mapping=slot_mapping,
        )
        return

    # 原本的实现（如果是非int8 kv cache，并且FA可用，则调用FA的reshape_and_cache；否则不做任何操作，假设上游已经正确更新了kv cache）
    if is_flash_attn_varlen_func_available():
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        return

    # fallback: do nothing
    return
```

### **PagedAttention**

```python
# 简易的pagedattention
def paged_attention_pytorch(
    q: torch.Tensor,                         # [total_q, Hq, D]
    key_cache: torch.Tensor,                 # [num_blocks, block_size, Hkv, D] int8 or fp
    value_cache: torch.Tensor,               # same
    cu_seqlens_q: torch.Tensor,              # [B+1]
    block_table: torch.Tensor,               # [B, max_blocks]
    seqused_k: torch.Tensor,                 # [B]  (kv length in tokens)
    softmax_scale: float,
    causal: bool,
    k_descale: Optional[torch.Tensor],        # [B, Hkv] if cache int8 else None
    v_descale: Optional[torch.Tensor],        # [B, Hkv] if cache int8 else None
    out: torch.Tensor,
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    """
    Simple paged attention (PyTorch), matching vLLM paged-kv semantics.
    Assumptions for simplicity:
      - Hq == Hkv (no GQA/MQA). If you need GQA, add head broadcast.
      - seqused_k gives total kv tokens available for each sequence.
    """
    device = q.device
    B = cu_seqlens_q.numel() - 1
    Hq = q.shape[1]
    D = q.shape[2]
    compute_dtype = _compute_dtype_for(q)

    # 直接往out里写
    for i in range(B):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        q_seq = q[q_start:q_end].to(compute_dtype)  # [q_len, H, D]
        q_len = q_seq.shape[0]

        kv_len = int(seqused_k[i].item())
        if kv_len <= 0 or q_len == 0:
            out[q_start:q_end].zero_()
            continue
				
        # 从block_table中获取当前序列已缓存的Block索引
        bt = block_table[i]  # [max_blocks]
        blocks = bt[bt != -1]
        if blocks.numel() == 0:
            out[q_start:q_end].zero_()
            continue

        # 从KV Cache中读取已缓存的Block（无需重新计算KV）
        # 这一步就是cache hit：直接从key_cache/value_cache读取已有Block，而非重新计算K/V
        kb = key_cache[blocks]
        vb = value_cache[blocks]

        # flatten tokens: [nb*block_size, Hkv, D] and truncate
        k_flat = kb.reshape(-1, kb.shape[-2], kb.shape[-1])[:kv_len]
        v_flat = vb.reshape(-1, vb.shape[-2], vb.shape[-1])[:kv_len]

        # dequant if int8
        if k_flat.dtype == torch.int8:
            assert k_descale is not None and v_descale is not None
            ks = k_descale[i].to(compute_dtype).view(1, -1, 1)  # [1,H,1]
            vs = v_descale[i].to(compute_dtype).view(1, -1, 1)
            k_seq = k_flat.to(compute_dtype) * ks
            v_seq = v_flat.to(compute_dtype) * vs
        else:
            k_seq = k_flat.to(compute_dtype)
            v_seq = v_flat.to(compute_dtype)

        # [q_len,H,D] -> [H,q_len,D]
        qh = q_seq.permute(1, 0, 2)
        kh = k_seq.permute(1, 0, 2)
        vh = v_seq.permute(1, 0, 2)

        assert qh.shape[0] == kh.shape[0], "This simple impl assumes Hq == Hkv."

        # 计算qk
        scores = torch.matmul(qh, kh.transpose(1, 2)) * softmax_scale  # [H,q,k]

        # causal mask (decoder-like, align q to the tail of kv)
        if causal:
            # key positions [0..kv_len-1], query positions correspond to [kv_len-q_len .. kv_len-1]
            q_pos = torch.arange(q_len, device=device)
            max_k = (kv_len - q_len) + q_pos  # [q_len]
            k_pos = torch.arange(kv_len, device=device).view(1, -1)  # [1,kv]
            mask = k_pos > max_k.view(-1, 1)  # [q_len,kv]
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask.unsqueeze(0), min_val)

        if softcap and softcap > 0:
            scores = (scores / softcap).tanh() * softcap

        # safe softmax
        scores = scores - scores.max(dim=-1, keepdim=True).values
        w = F.softmax(scores, dim=-1)
        w = torch.nan_to_num(w, 0.0, 0.0, 0.0)

        out_h = torch.matmul(w, vh)  # [H,q,D]
        out_seq = out_h.permute(1, 0, 2).to(q.dtype)  # [q,H,D]
        out[q_start:q_end] = out_seq

    return out
```

### **reshape_and_cache_flash_pytorch**

```python
def reshape_and_cache_flash_pytorch(
    key_int8: torch.Tensor,        # [T, Hkv, D] (already int8)
    value_int8: torch.Tensor,      # [T, Hkv, D] (already int8)
    key_cache_i8: torch.Tensor,    # [num_blocks, block_size, Hkv, D] int8 view
    value_cache_i8: torch.Tensor,  # same
    slot_mapping: torch.Tensor,    # [T], slot = block_id * block_size + token_in_block, or -1 相当于cache的idx
) -> None:
    """
    把已经量化成 INT8 的 K/V 序列（key_int8/value_int8），
    按照slot_mapping(idx)指定的内存位置，写入到分页管理的 INT8 KV Cache（key_cache_i8/value_cache_i8）中。
    """
    slot_mapping = slot_mapping.flatten()
    valid = slot_mapping >= 0
    if valid.sum().item() == 0:
        return

    slots = slot_mapping[valid].to(torch.int64)
    k = key_int8[valid]      # [N, Hkv, D]
    v = value_int8[valid]

    block_size = key_cache_i8.shape[1]
    block_idx = slots // block_size
    tok_idx = slots % block_size

    with torch.no_grad():
        key_cache_i8[block_idx, tok_idx] = k
        value_cache_i8[block_idx, tok_idx] = v
```

Repo:

```
git clone https://github.com/haiyu-wei/vllm_in8kvcache.git
cd vllm_in8kvcache
python test_int8_paged.py 
```










<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
