import copy
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

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