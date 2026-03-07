import copy
from dataclasses import dataclass
import math
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange

def flash_attn_varlen_func_pytorch(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=0,
    fa_version=2,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """
    flash_attn_varlen_func 的PyTorch实现。
    
    此函数提供与vLLM flash attention接口兼容的纯PyTorch实现。
    
    Args:
        q: (total_q_tokens, num_heads, head_dim)
        k: (num_blocks, block_size, num_kv_heads, head_dim) - KV缓存
        v: (num_blocks, block_size, num_kv_heads, head_dim) - KV缓存
        max_seqlen_q: int - 最大查询序列长度
        cu_seqlens_q: (batch_size + 1,) - 查询的累积序列长度
        max_seqlen_k: int - 最大键序列长度
        cu_seqlens_k: 可选 - 键的累积序列长度（用于非分页预填充）
        seqused_k: (batch_size,) - 实际使用的键序列长度
        q_v: 可选 - 查询的值投影（此实现中未使用）
        dropout_p: float - dropout概率
        softmax_scale: float - 注意力分数的缩放因子
        causal: bool - 是否应用因果掩码
        window_size: 滑动窗口注意力的可选元组
        softcap: float - 注意力分数的软上限值
        alibi_slopes: ALiBi偏置的可选张量
        deterministic: bool - 是否使用确定性dropout
        return_attn_probs: bool - 是否返回注意力概率
        block_table: (batch_size, max_blocks_per_seq) - 从序列到块的映射
        return_softmax_lse: bool - 是否返回softmax logsumexp
        out: 可选的输出张量
        scheduler_metadata: 调度器元数据（在PyTorch实现中忽略）
        q_descale, k_descale, v_descale: 量化输入的反缩放因子
        num_splits: 注意力计算的分割数
        fa_version: flash attention版本（在PyTorch实现中忽略）
        s_aux: 辅助sink tokens
        cp_world_size: 跨进程世界大小
        cp_rank: 跨进程排名
        cp_tot_seqused_k: 跨进程键的总序列使用量
    
    Returns:
        output: (total_q_tokens, num_heads, head_dim)
        如果 return_attn_probs=True: 同时返回注意力概率
        如果 return_softmax_lse=True: 同时返回softmax logsumexp
    """
    device = q.device
    dtype = q.dtype
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[2]
    
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    if window_size is None:
        window_size = (-1, -1)
    
    # 检查是否为连续KV数据模式（k的形状不是分页格式）
    is_continuous_kv = len(k.shape) == 3  # (total_tokens, num_heads, head_dim)
    
    # 只有在分页KV数据模式下才需要block_table
    if not is_continuous_kv and block_table is None:
        raise ValueError("block_table is required for paged KV cache")
    
    # 在分页模式下获取block_size，连续模式下使用默认值
    if not is_continuous_kv:
        block_size = k.shape[1]
    else:
        block_size = 128  # 连续数据模式下使用默认block_size
    
    q = q.float()
    k = k.float()
    v = v.float()
    
    # 检查是否为连续KV数据模式（k的形状不是分页格式）
    is_continuous_kv = len(k.shape) == 3  # (total_tokens, num_heads, head_dim)
    
    if is_continuous_kv:
        # 连续KV数据模式：k的形状是(total_tokens, num_heads, head_dim)
        # 需要正确设置k_batch和v_batch的维度
        num_heads_k = k.shape[1]  # 从输入数据获取实际的head数量
        k_batch = torch.zeros(batch_size, max_seqlen_k, num_heads_k, head_dim, dtype=torch.float32, device=device)
        v_batch = torch.zeros(batch_size, max_seqlen_k, num_heads_k, head_dim, dtype=torch.float32, device=device)
        
        # 连续KV数据模式：直接使用cu_seqlens_k索引
        for batch_idx in range(batch_size):
            k_start = cu_seqlens_k[batch_idx].item()
            k_end = cu_seqlens_k[batch_idx + 1].item()
            seq_len_k = k_end - k_start
            
            if seq_len_k > 0:
                k_batch[batch_idx, :seq_len_k] = k[k_start:k_end]
                v_batch[batch_idx, :seq_len_k] = v[k_start:k_end]
    else:
        # 分页KV数据模式：使用原来的维度设置
        k_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, dtype=torch.float32, device=device)
        v_batch = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, dtype=torch.float32, device=device)
        
        # 分页KV数据模式：使用block_table索引
        for batch_idx in range(batch_size):
            seq_len = seqused_k[batch_idx].item() if seqused_k is not None else max_seqlen_k
            blocks = block_table[batch_idx]
            
            for token_idx in range(seq_len):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                
                # 检查block_idx是否在blocks数组范围内
                if block_idx < len(blocks):
                    actual_block_idx = blocks[block_idx].item()
                    
                    if actual_block_idx >= 0 and actual_block_idx < k.shape[0]:
                        k_batch[batch_idx, token_idx] = k[actual_block_idx, block_offset]
                        v_batch[batch_idx, token_idx] = v[actual_block_idx, block_offset]
    
    q_batch = torch.zeros(batch_size, max_seqlen_q, num_heads, head_dim, dtype=torch.float32, device=device)
    
    # 处理查询序列
    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        seq_len_q = q_end - q_start
        if seq_len_q > 0:
            q_batch[batch_idx, :seq_len_q] = q[q_start:q_end]
    
    query_padding_mask = torch.zeros(batch_size, max_seqlen_q, dtype=torch.bool, device=device)
    key_padding_mask = torch.zeros(batch_size, max_seqlen_k, dtype=torch.bool, device=device)
    
    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        seq_len_q = q_end - q_start
        seq_len_k = seqused_k[batch_idx].item() if seqused_k is not None else max_seqlen_k
        
        if seq_len_q > 0:
            query_padding_mask[batch_idx, :seq_len_q] = True
        if seq_len_k > 0:
            key_padding_mask[batch_idx, :seq_len_k] = True
    
    attn_bias = None
    if alibi_slopes is not None:
        row_idx = torch.arange(max_seqlen_q, device=device, dtype=torch.float32).unsqueeze(1)
        col_idx = torch.arange(max_seqlen_k, device=device, dtype=torch.float32)
        if causal:
            attn_bias = (col_idx - row_idx) * alibi_slopes.unsqueeze(-1).unsqueeze(-1)
        else:
            attn_bias = -torch.abs(row_idx - col_idx) * alibi_slopes.unsqueeze(-1).unsqueeze(-1)
    
    # 确保k_batch和v_batch的头数正确扩展到匹配q_batch
    num_heads_q = q_batch.shape[2]
    num_heads_k = k_batch.shape[2]
    if num_heads_q != num_heads_k:
        g = num_heads_q // num_heads_k
        k_batch = repeat(k_batch, "b s h d -> b s (h g) d", g=g)
        v_batch = repeat(v_batch, "b s h d -> b s (h g) d", g=g)
    
    output, attention = attention_ref(
        q=q_batch,
        k=k_batch,
        v=v_batch,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        dropout_mask=None,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=False,
        key_leftpad=None,
    )
    
    # 重组输出
    outputs = []
    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        seq_len_q = q_end - q_start
        # output shape: [batch, seq, heads, dim] -> 需要提取正确的序列部分
        batch_output = output[batch_idx, :seq_len_q]  # [seq_len_q, heads, dim]
        outputs.append(batch_output)
    
    final_output = torch.cat(outputs, dim=0)  # [total_seq, heads, dim]
    final_output = final_output.to(dtype)
    
    if out is not None:
        out.copy_(final_output)
        final_output = out
    
    result = [final_output]
    
    if return_softmax_lse:
        softmax_lse = torch.logsumexp(attention, dim=-1)
        result.append(softmax_lse)
    
    if return_attn_probs:
        result.append(attention)
    
    return tuple(result) if len(result) > 1 else result[0]


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


# -------- Int8 dynamic quant path: gather KV, quantize Q/K/V, simulate int8 attention --------


def gather_kv_from_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K, V from paged cache into contiguous [total_kv, num_kv_heads, head_dim]."""
    # key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    batch_size = block_table.shape[0]
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    device = key_cache.device
    dtype = key_cache.dtype

    total_kv = seqused_k.sum().item()
    k_contig = torch.zeros(
        (total_kv, num_kv_heads, head_dim), dtype=dtype, device=device
    )
    v_contig = torch.zeros(
        (total_kv, num_kv_heads, head_dim), dtype=dtype, device=device
    )

    offset = 0
    for batch_idx in range(batch_size):
        seq_len = seqused_k[batch_idx].item()
        blocks = block_table[batch_idx]
        for token_idx in range(seq_len):
            block_idx = token_idx // block_size
            block_offset = token_idx % block_size
            if block_idx < blocks.shape[0]:
                physical_block = blocks[block_idx].item()
                if physical_block >= 0:
                    k_contig[offset] = key_cache[physical_block, block_offset]
                    v_contig[offset] = value_cache[physical_block, block_offset]
            offset += 1
    return k_contig, v_contig


def dynamic_quantize_to_int8_per_head(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize x to int8 with per-head scale (scale = max(abs(x)) / 127 for each head).
    x: [..., num_heads, head_dim] or [..., num_kv_heads, head_dim].
    Returns (x_int8, scale) where scale is [num_heads] or [num_kv_heads].
    """
    # x: [N, H, D] -> per-head scale over all tokens and head_dim, shape (h,)
    n, h, d = x.shape[0], x.shape[1], x.shape[2]
    x_flat = x.float().view(n, h, -1)
    scale = x_flat.abs().amax(dim=(0, -1), keepdim=False).clamp(min=1e-8) / 127.0
    x_scaled = (x.float() / scale.unsqueeze(0).unsqueeze(-1)).round().clamp(
        -128, 127
    )
    x_int8 = x_scaled.to(torch.int8)
    return x_int8, scale.float()


def int8_attention_varlen_simulate(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None = None,
    window_size: tuple[int | None, int | None] = (-1, -1),
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    """
    Simulate int8 attention: no dequant of Q/K/V before matmul.
    - scores_i32 = Q_i8 @ K_i8^T, then logits = scores_i32 * (q_scale * k_scale / sqrt(d))
    - softmax(logits)
    - output = (attn @ V_i8) * v_scale
    Q/K/V are int8; scales applied only after integer matmul to mimic real int8 kernel.
    """
    device = q_int8.device
    # K/V are contiguous [total_kv, num_kv_heads, head_dim] with batch 0, then 1, ...
    total_kv = k_int8.shape[0]
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    offset = 0
    for b in range(batch_size):
        cu_seqlens_k[b] = offset
        offset += seqused_k[b].item()
    cu_seqlens_k[batch_size] = offset

    q_batch = torch.zeros(
        batch_size, max_seqlen_q, num_heads, head_dim,
        dtype=torch.float32, device=device
    )
    k_batch = torch.zeros(
        batch_size, max_seqlen_k, num_kv_heads, head_dim,
        dtype=torch.float32, device=device
    )
    v_batch = torch.zeros(
        batch_size, max_seqlen_k, num_kv_heads, head_dim,
        dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        sq, sk = q_end - q_start, k_end - k_start
        if sq > 0:
            q_batch[b, :sq] = q_int8[q_start:q_end].float()
        if sk > 0:
            k_batch[b, :sk] = k_int8[k_start:k_end].float()
            v_batch[b, :sk] = v_int8[k_start:k_end].float()

    if num_heads != num_kv_heads:
        g = num_heads // num_kv_heads
        k_batch = repeat(k_batch, "b s h d -> b s (h g) d", g=g)
        v_batch = repeat(v_batch, "b s h d -> b s (h g) d", g=g)

    # Simulate int8: matmul with int8 values (no dequant before matmul), then scale result
    # scores = (Q_i8 @ K_i8^T) * (q_scale * k_scale / sqrt(d))
    d = head_dim
    n_heads_q = q_batch.shape[2]
    n_heads_kv = k_batch.shape[2]
    scores = torch.einsum("bthd,bshd->bhts", q_batch, k_batch)
    q_scale_ = q_scale.view(-1).float().to(device)
    k_scale_ = k_scale.view(-1).float().to(device)
    if q_scale_.numel() == 1:
        q_scale_ = q_scale_.expand(n_heads_q)
    if k_scale_.numel() == 1:
        k_scale_ = k_scale_.expand(n_heads_kv)
    if n_heads_q != n_heads_kv:
        k_scale_ = k_scale_.repeat(n_heads_q // n_heads_kv)
    scale_scores = (
        q_scale_.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        * k_scale_.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        * softmax_scale
    )
    scores = scores.float() * scale_scores.to(scores.device)
    if logits_soft_cap > 0:
        scores = (scores / logits_soft_cap).tanh() * logits_soft_cap
    if alibi_slopes is not None:
        row = torch.arange(max_seqlen_q, device=device, dtype=torch.float32)
        col = torch.arange(max_seqlen_k, device=device, dtype=torch.float32)
        if causal:
            bias = (col - row).unsqueeze(0).unsqueeze(0) * alibi_slopes.view(
                1, -1, 1, 1
            )
        else:
            bias = -torch.abs(col - row).unsqueeze(0).unsqueeze(0) * alibi_slopes.view(1, -1, 1, 1)
        scores = scores + bias.to(scores.dtype)

    if causal and max_seqlen_q > 1:
        # Prefill: mask future keys (col > row). Decode (max_seqlen_q==1): single query
        # must attend to all keys, so do not mask.
        mask = torch.arange(max_seqlen_k, device=device).unsqueeze(0) > torch.arange(
            max_seqlen_q, device=device
        ).unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    for b in range(batch_size):
        sk = seqused_k[b].item()
        if sk < max_seqlen_k:
            scores[b, :, :, sk:] = float("-inf")

    attn = F.softmax(scores, dim=-1)
    # Simulate int8: (attn @ V_i8) then * v_scale, no dequant of V before matmul
    out = torch.einsum("bhts,bshd->bthd", attn, v_batch)
    v_scale_ = v_scale.view(-1).float().to(device)
    if v_scale_.numel() == 1:
        v_scale_ = v_scale_.expand(out.shape[2])
    elif v_scale_.numel() == num_kv_heads and out.shape[2] != num_kv_heads:
        v_scale_ = v_scale_.repeat(out.shape[2] // num_kv_heads)
    out = out.float() * v_scale_.view(1, 1, -1, 1).to(out.device)
    out_flat = []
    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        sq = q_end - q_start
        if sq > 0:
            out_flat.append(out[b, :sq])
    return torch.cat(out_flat, dim=0)