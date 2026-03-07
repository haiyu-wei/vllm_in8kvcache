import copy
from dataclasses import dataclass
import math
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange

def _compute_dtype_for(q: torch.Tensor) -> torch.dtype:
    # bfloat16 is safer in fp32 for stability
    return torch.float32 if q.dtype == torch.bfloat16 else torch.float16

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

        bt = block_table[i]  # [max_blocks]
        blocks = bt[bt != -1]
        if blocks.numel() == 0:
            out[q_start:q_end].zero_()
            continue

        # gather blocks: [nb, block_size, Hkv, D]
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
        print(f"DTYPES: qh={qh.dtype}, kh={kh.dtype}, scores={scores.dtype}")

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



def reshape_and_cache_flash_pytorch(
    key_int8: torch.Tensor,        # [T, Hkv, D] (already int8)
    value_int8: torch.Tensor,      # [T, Hkv, D] (already int8)
    key_cache_i8: torch.Tensor,    # [num_blocks, block_size, Hkv, D] int8 view
    value_cache_i8: torch.Tensor,  # same
    slot_mapping: torch.Tensor,    # [T], slot = block_id * block_size + token_in_block, or -1 相当于cache的idx
) -> None:
    """
    把已经量化成 INT8 的 K/V 序列（key_int8/value_int8），
    按照slot_mapping指定的内存位置，写入到分页管理的 INT8 KV Cache（key_cache_i8/value_cache_i8）中。
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

def dynamic_quantize_tensor_per_head(x, is_kv_cache=False):
    """
    动态对称量化 (Per-Head 版本)。
    
    Args:
        x: 输入张量
           - Q 的形状: [num_tokens, num_heads, head_dim] (3D)
           - K/V Cache 的形状: [num_blocks, block_size, num_kv_heads, head_dim] (4D)
        is_kv_cache: 输入是否为 4D 的 KV Cache
    
    Returns:
        x_int8: 量化后的 INT8 张量
        scale: Per-Head 的 scale 张量
               - Q: [1, num_heads, 1] (方便广播)
               - KV Cache: [1, 1, num_kv_heads, 1]
    """
    # 1. 强制转换为 FP32 以确保计算精度
    x_fp32 = x.float()
    
    # 2. 确定在哪些维度上求最大值 (保留 Head 维度)
    # 目标：对于每个 Head，算出它的最大值
    if is_kv_cache:
        # KV Cache 是 4D: [Blocks, BlockSize, Heads, Dim]
        # 我们要在 dim=0, 1, 3 上求 max，保留 dim=2 (Heads)
        dims_to_reduce = (0, 1, 3)
        # 为了后续广播，scale 需要变成 [1, 1, num_kv_heads, 1]
        scale_shape = (1, 1, x_fp32.size(2), 1)
    else:
        # Q/K/V 是 3D: [Tokens, Heads, Dim]
        # 我们要在 dim=0, 2 上求 max，保留 dim=1 (Heads)
        dims_to_reduce = (0, 2)
        # 为了后续广播，scale 需要变成 [1, num_heads, 1]
        scale_shape = (1, x_fp32.size(1), 1)

    # 3. 计算 Per-Head 的绝对值最大值
    # keepdim=True 非常重要，保留维度以便广播
    abs_max = x_fp32.abs().amax(dim=dims_to_reduce, keepdim=True)
    
    # 4. 计算 Scale (映射到 [-127, 127])
    scale = abs_max / 127.0
    scale = torch.clamp(scale, min=1e-9) # 防止除零
    
    # 5. 执行量化
    # 利用广播机制：x_fp32 / scale 会自动对齐维度
    x_int8 = torch.round(x_fp32 / scale).clamp(-128, 127).to(torch.int8)
    
    # 6. 把 scale 展平成 vLLM FA 需要的形状 (可选，视后续接口需求)
    # vLLM 通常需要 [batch, num_heads] 的 2D 形状
    # 这里我们先返回带 keepdim 的版本，方便在 forward 里 reshape
    
    return x_int8, scale

def dynamic_quantize_tensor(x_fp32):
    """
    动态对称量化：将 FP32 张量量化为 INT8。
    公式： q = round(x / scale)
    """
    print(f"动态量化输入张量的 dtype: {x_fp32.dtype}, shape: {x_fp32.shape}")
    print(f"x_fp32 统计: min={x_fp32.min().item():.4f}, max={x_fp32.max().item():.4f}, mean={x_fp32.mean().item():.4f}")
    # 计算缩放因子：使用绝对值最大值映射到 INT8 范围 (-127, 127)
    # 注意：保留 -128 以避免溢出风险，或者直接用 127.0
    abs_max = x_fp32.abs().max()
    scale = abs_max / 127.0 
    
    # 为了数值稳定性，如果全零则 scale 设为 1
    scale = torch.clamp(scale, min=1e-9) 
    
    # 量化
    x_int8 = torch.round(x_fp32 / scale).to(torch.int8)
    print(f"量化完成: scale={scale.item():.6f}, x_int8 dtype: {x_int8.dtype}, shape: {x_int8.shape}")
    print(f"x_int8 统计: min={x_int8.min().item()}, max={x_int8.max().item()}, mean={x_int8.float().mean().item():.4f}")


    return x_int8, scale

# =============================================================================
# 3. 核心实现：支持 INT8 的 PyTorch Flash Attention Varlen
# =============================================================================

def flash_attn_varlen_func_pytorch_int8(
    q, k, v,
    max_seqlen_q, cu_seqlens_q, max_seqlen_k,
    cu_seqlens_k=None, seqused_k=None, q_v=None,
    dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=None, softcap=0.0, alibi_slopes=None,
    deterministic=False, return_attn_probs=False,
    block_table=None, return_softmax_lse=False,
    out=None, scheduler_metadata=None,
    q_descale=None, k_descale=None, v_descale=None, # INT8 反缩放因子
    num_splits=0, fa_version=2, s_aux=None,
    **kwargs
):
    device = q.device
    dtype = q.dtype
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    if window_size is None:
        window_size = (-1, -1)

    # --------------------------
    # 关键修改：INT8 反量化入口
    # --------------------------
    # def maybe_dequantize(x, descale, name):
    #     print(f"输入{name}的 dtype: {x.dtype}, shape: {x.shape}")
    #     if x.dtype == torch.int8:
    #         if descale is None:
    #             raise ValueError(f"Input {name} is INT8 but {name}_descale is None!")
    #         # 反量化回 FP32 进行计算
    #         return x.to(torch.float32) * descale
    #     return x.float()
    def maybe_dequantize(x, descale, name):
        print(f"输入{name}的 dtype: {x.dtype}, shape: {x.shape}")
        if x.dtype == torch.int8:
            if descale is None:
                raise ValueError(f"Input {name} is INT8 but {name}_descale is None!")
            
            x_fp32 = x.to(torch.float32)
            
            # ---------------- 核心修复：形状广播适配 ----------------
            # 确保 descale 是一个 tensor
            if not torch.is_tensor(descale):
                descale = torch.tensor(descale, device=x.device, dtype=torch.float32)
                
            # 分析 x 的形状，自动适配 descale 的维度
            # 情况 1: x 是 3D (例如 Query: [TotalTokens, NumHeads, HeadDim])
            if x.dim() == 3:
                # descale 可能是 [] (标量) 或 [NumHeads]
                # 将 descale 变为 [1, *, 1] 以便广播
                if descale.dim() == 0:
                    # 标量，无需处理
                    pass 
                elif descale.dim() == 1:
                    # Per-Head: [Heads] -> [1, Heads, 1]
                    descale = descale.view(1, -1, 1)
            
            # 情况 2: x 是 4D (例如 Paged KV Cache: [Blocks, BlockSize, NumKVHeads, HeadDim])
            elif x.dim() == 4:
                # descale 可能是 [] (标量) 或 [NumKVHeads]
                if descale.dim() == 0:
                    pass
                elif descale.dim() == 1:
                    # Per-Head: [Heads] -> [1, 1, Heads, 1]
                    descale = descale.view(1, 1, -1, 1)
            
            # 执行反量化
            return x_fp32 * descale
            # --------------------------------------------------------
            
        return x.float()

    q_fp32 = maybe_dequantize(q, q_descale, "q")
    k_fp32 = maybe_dequantize(k, k_descale, "k")
    v_fp32 = maybe_dequantize(v, v_descale, "v")
    print(f"dequantized: q_fp32 dtype: {q_fp32.dtype}, k_fp32 dtype: {k_fp32.dtype}, v_fp32 dtype: {v_fp32.dtype}")
    print(f"q_fp32 统计: min={q_fp32.min().item():.4f}, max={q_fp32.max().item():.4f}, mean={q_fp32.mean().item():.4f}"
          f"\nk_fp32 统计: min={k_fp32.min().item():.4f}, max={k_fp32.max().item():.4f}, mean={k_fp32.mean().item():.4f}"
            f"\nv_fp32 统计: min={v_fp32.min().item():.4f}, max={v_fp32.max().item():.4f}, mean={v_fp32.mean().item():.4f}")
    
    # 判断 KV 格式
    is_continuous_kv = len(k.shape) == 3
    if not is_continuous_kv and block_table is None:
        raise ValueError("block_table required for paged KV")
    
    block_size = k.shape[1] if not is_continuous_kv else 128

    # --------------------------
    # 数据重组 (Reformatting)
    # --------------------------
    
    # 1. 重组 K
    if is_continuous_kv:
        num_kv_heads = k_fp32.shape[1]
        k_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=torch.float32)
        v_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=torch.float32)
        for b in range(batch_size):
            start = cu_seqlens_k[b].item()
            end = cu_seqlens_k[b+1].item()
            if end > start:
                k_batch[b, :end-start] = k_fp32[start:end]
                v_batch[b, :end-start] = v_fp32[start:end]
    else:
        num_kv_heads = k_fp32.shape[2]
        k_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=torch.float32)
        v_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=torch.float32)
        for b in range(batch_size):
            seq_len = seqused_k[b].item() if seqused_k is not None else max_seqlen_k
            blocks = block_table[b]
            for t in range(seq_len):
                bid = blocks[t // block_size].item()
                off = t % block_size
                if 0 <= bid < k_fp32.shape[0]:
                    k_batch[b, t] = k_fp32[bid, off]
                    v_batch[b, t] = v_fp32[bid, off]

    # 2. 重组 Q
    q_batch = torch.zeros(batch_size, max_seqlen_q, num_heads, head_dim, device=device, dtype=torch.float32)
    query_padding_mask = torch.zeros(batch_size, max_seqlen_q, dtype=torch.bool, device=device)
    key_padding_mask = torch.zeros(batch_size, max_seqlen_k, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b+1].item()
        len_q = q_end - q_start
        len_k = seqused_k[b].item() if seqused_k is not None else max_seqlen_k
        
        if len_q > 0:
            q_batch[b, :len_q] = q_fp32[q_start:q_end]
            query_padding_mask[b, :len_q] = True
        if len_k > 0:
            key_padding_mask[b, :len_k] = True

    # --------------------------
    # 注意力计算
    # --------------------------
    from einops import repeat, rearrange
    # MQA/GQA 头扩展
    if q_batch.shape[2] != k_batch.shape[2]:
        g = q_batch.shape[2] // k_batch.shape[2]
        k_batch = repeat(k_batch, "b s h d -> b s (h g) d", g=g)
        v_batch = repeat(v_batch, "b s h d -> b s (h g) d", g=g)

    output, attention = attention_ref(
        q=q_batch, k=k_batch, v=v_batch,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=causal, window_size=window_size,
        softcap=softcap, upcast=True
    )

    # --------------------------
    # 输出还原
    # --------------------------
    outputs = []
    for b in range(batch_size):
        start = cu_seqlens_q[b].item()
        end = cu_seqlens_q[b+1].item()
        outputs.append(output[b, :end-start])
    
    final_output = torch.cat(outputs, dim=0)
    
    # 注意：如果输入是 INT8，输出通常保持 FP32 以保留精度，除非后续有后量化
    # 这里我们默认输出回输入的原始 dtype (如果是 FP16/BF16) 或者保持 FP32 (如果输入是 INT8)
    if dtype != torch.int8:
        final_output = final_output.to(dtype)
    else:
        final_output = final_output.to(torch.float32) # INT8 输入 -> FP32 输出是行业惯例

    if out is not None:
        out.copy_(final_output)
        final_output = out

    return final_output



def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    # 计算注意力分数
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    if key_padding_mask is not None:
        output.masked_fill_(rearrange(torch.logical_not(torch.any(key_padding_mask, 1)), "b -> b 1 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

