import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def flash_attn_varlen_func_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: float = 1.0,
    causal: bool = False,
    alibi_slopes: Optional[torch.Tensor] = None,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: Optional[float] = None,
    fa_version: Optional[int] = None,  # 仅用于兼容原接口，PyTorch版忽略
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    num_splits: int = 0,  # 仅用于兼容原接口，PyTorch版忽略
) -> torch.Tensor:
    """
    纯PyTorch重写的变长Flash Attention算子，功能等价于原CUDA版本
    核心参数说明（与原函数一致）：
    - q/k/v: [total_q/k/v_tokens, num_heads, head_dim]，展平的变长序列张量
    - cu_seqlens_q/k: [batch_size+1]，累积序列长度（指示每个序列的起始/结束索引）
    - q_descale/k_descale/v_descale: 量化反量化缩放因子
    - alibi_slopes: [num_heads]，ALiBi注意力偏置的斜率
    - window_size: (left, right)，滑动窗口大小（-1表示无限制）
    - softcap: logits软上限
    """
    # ---------------- 1. 预处理：基础参数与设备对齐 ----------------
    device = q.device
    total_q, num_heads, head_dim = q.shape
    total_kv = k.shape[0]
    batch_size = cu_seqlens_q.shape[0] - 1 if cu_seqlens_q is not None else 1

    # ---------------- 2. 量化反量化（如果提供了descale） ----------------
    if q_descale is not None:
        q = q * q_descale.to(device)
    if k_descale is not None:
        k = k * k_descale.to(device)
    if v_descale is not None:
        v = v * v_descale.to(device)

    # ---------------- 3. 初始化输出张量 ----------------
    if out is None:
        out = torch.empty_like(q)

    # ---------------- 4. 核心：逐序列计算变长注意力 ----------------
    for i in range(batch_size):
        # 4.1 提取当前序列的Q/K/V（根据cu_seqlens）
        if cu_seqlens_q is not None:
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i+1].item()
            q_seq = q[q_start:q_end]  # [q_len, num_heads, head_dim]
        else:
            q_seq = q  # 单序列场景
        
        if cu_seqlens_k is not None:
            k_start = cu_seqlens_k[i].item()
            k_end = cu_seqlens_k[i+1].item()
            k_seq = k[k_start:k_end]  # [k_len, num_heads, head_dim]
            v_seq = v[k_start:k_end]  # [k_len, num_heads, head_dim]
        else:
            k_seq = k
            v_seq = v
        
        q_len = q_seq.shape[0]
        k_len = k_seq.shape[0]

        # 4.2 调整维度以适配PyTorch矩阵乘法
        # [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q_seq = q_seq.permute(1, 0, 2)
        k_seq = k_seq.permute(1, 0, 2)
        v_seq = v_seq.permute(1, 0, 2)

        # 4.3 计算注意力分数：Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(q_seq, k_seq.transpose(1, 2))  # [num_heads, q_len, k_len]
        attn_scores = attn_scores * softmax_scale  # 应用softmax_scale

        # ---------------- 5. 可选功能：滑动窗口 ----------------
        if window_size is not None and window_size != (-1, -1):
            left, right = window_size
            # 生成窗口掩码：只保留[q_pos-left, q_pos+right]范围内的位置
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
            for q_pos in range(q_len):
                start = max(0, q_pos - left)
                end = min(k_len, q_pos + right + 1)
                mask[q_pos, start:end] = False
            # 应用掩码（将窗口外的分数设为-inf）
            attn_scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

        # ---------------- 6. 可选功能：ALiBi注意力偏置 ----------------
        if alibi_slopes is not None:
            # 生成ALiBi偏置矩阵
            alibi_slopes = alibi_slopes.to(device)
            # 计算位置差：[q_len, k_len]
            q_pos = torch.arange(q_len, device=device).unsqueeze(1)
            k_pos = torch.arange(k_len, device=device).unsqueeze(0)
            pos_diff = q_pos - k_pos  # [q_len, k_len]
            # 生成ALiBi偏置：[num_heads, q_len, k_len]
            alibi_bias = -alibi_slopes.unsqueeze(1).unsqueeze(2) * pos_diff.abs().unsqueeze(0)
            # 应用ALiBi偏置
            attn_scores = attn_scores + alibi_bias

        # ---------------- 7. 可选功能：因果掩码（Encoder场景causal=False，可选） ----------------
        if causal:
            # 生成下三角掩码（只看过去和当前位置）
            mask = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=device), diagonal=1)
            attn_scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

        # ---------------- 8. 可选功能：logits softcap ----------------
        if softcap is not None and softcap > 0:
            # 应用软上限：tanh(logits / softcap) * softcap
            attn_scores = attn_scores / softcap
            attn_scores = attn_scores.tanh()
            attn_scores = attn_scores * softcap

        # ---------------- 9. Softmax归一化 + 计算最终输出 ----------------
        attn_weights = F.softmax(attn_scores, dim=-1)  # [num_heads, q_len, k_len]
        output_seq = torch.matmul(attn_weights, v_seq)  # [num_heads, q_len, head_dim]

        # ---------------- 10. 调整维度并写回输出张量 ----------------
        # [num_heads, q_len, head_dim] -> [q_len, num_heads, head_dim]
        output_seq = output_seq.permute(1, 0, 2)
        if cu_seqlens_q is not None:
            out[q_start:q_end] = output_seq
        else:
            out = output_seq

    return out
import torch
from typing import Optional

def reshape_and_cache_flash_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    """
    最终修复版：基于 slot_mapping = block_idx * 16 + token_idx_in_block 的逻辑
    """
    device = key.device
    slot_mapping = slot_mapping.flatten()

    # ---------------- 1. 自动适配输入维度 ----------------
    if len(key.shape) == 4:
        batch_size, num_heads, seq_len, head_dim = key.shape
        key_flat = key.reshape(-1, num_heads, head_dim)
        value_flat = value.reshape(-1, num_heads, head_dim)
    elif len(key.shape) == 3:
        total_tokens, num_heads, head_dim = key.shape
        key_flat = key
        value_flat = value
    else:
        raise ValueError(f"不支持的key维度: {len(key.shape)}")

    # ---------------- 2. 数据类型检查 ----------------
    if kv_cache_dtype == "int8":
        assert key.dtype == torch.int8 and value.dtype == torch.int8
    elif kv_cache_dtype == "fp16":
        key_flat = key_flat.to(torch.float16)
        value_flat = value_flat.to(torch.float16)
    elif kv_cache_dtype == "fp32":
        key_flat = key_flat.to(torch.float32)
        value_flat = value_flat.to(torch.float32)

    # ---------------- 3. 核心逻辑：解码 slot_mapping 并写入 ----------------
    # 过滤有效槽位
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    if len(valid_slots) == 0:
        return

    # 提取有效 K/V
    valid_key = key_flat[valid_mask]
    valid_value = value_flat[valid_mask]

    # 解码 slot_mapping：slot = block_idx * 16 + token_idx_in_block
    tokens_per_block = 16
    block_indices = valid_slots // tokens_per_block
    token_indices = valid_slots % tokens_per_block

    # 直接使用高级索引写入（最简洁高效的方式）
    key_cache[block_indices, token_indices] = valid_key
    value_cache[block_indices, token_indices] = valid_value


# import torch
# from typing import Optional

# def reshape_and_cache_flash_pytorch(
#     key: torch.Tensor,
#     value: torch.Tensor,
#     key_cache: torch.Tensor,
#     value_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
#     kv_cache_dtype: str,
#     k_scale: Optional[torch.Tensor] = None,
#     v_scale: Optional[torch.Tensor] = None,
# ) -> None:
#     """
#     纯PyTorch重写的reshape_and_cache_flash函数
#     核心功能：
#     1. 将K/V重塑为Flash Attention兼容的KV Cache维度
#     2. 根据slot_mapping写入对应的Cache位置
#     3. 兼容INT8量化（带scale）和FP16/FP32格式
    
#     参数说明（与原函数完全一致）：
#     - key/value: [batch_size, num_heads, seq_len, head_dim]，输入的K/V张量
#     - key_cache/value_cache: Flash Attention格式的KV Cache
#     - slot_mapping: [batch_size, seq_len]，展平的槽位映射表
#     - kv_cache_dtype: "int8" / "fp16" / "fp32"
#     - k_scale/v_scale: INT8量化的缩放因子
#     """
#     # ---------------- 1. 预处理：基础参数与设备对齐 ----------------
#     """
#     纯PyTorch重写的reshape_and_cache_flash函数（修复版：兼容3维和4维输入）
#     支持的输入形状：
#     - 4维: [batch_size, num_heads, seq_len, head_dim]
#     - 3维: [total_tokens, num_heads, head_dim] (vLLM v1/变长序列常见)
#     """
#     # ---------------- 1. 预处理：自动适配输入维度 ----------------
#     device = key.device
#     key_shape = key.shape
    
#     # 自动判断并适配维度
#     if len(key_shape) == 4:
#         # 4维输入: [batch_size, num_heads, seq_len, head_dim]
#         batch_size, num_heads, seq_len, head_dim = key_shape
#         # 展平为3维: [batch_size*seq_len, num_heads, head_dim]
#         key_flat = key.reshape(-1, num_heads, head_dim)
#         value_flat = value.reshape(-1, num_heads, head_dim)
#     elif len(key_shape) == 3:
#         # 3维输入: [total_tokens, num_heads, head_dim] (vLLM v1常见)
#         total_tokens, num_heads, head_dim = key_shape
#         key_flat = key
#         value_flat = value
#     else:
#         raise ValueError(f"不支持的key维度: {len(key_shape)} (仅支持3维或4维)")

#     # ---------------- 2. 数据类型适配（量化/非量化） ----------------
#     if kv_cache_dtype == "int8":
#         # 确保输入已经是INT8（原函数输入是key_int8/value_int8）
#         assert key.dtype == torch.int8, "INT8模式下key必须是int8类型"
#         assert value.dtype == torch.int8, "INT8模式下value必须是int8类型"
#         # 保存量化scale（如果提供）
#         if k_scale is not None:
#             # 这里假设k_scale已经是适配Cache的维度，直接保存到对应位置
#             # 注意：原函数中scale通常保存在单独的张量中，这里简化处理
#             pass
#     elif kv_cache_dtype == "fp16":
#         key = key.to(torch.float16)
#         value = value.to(torch.float16)
#     elif kv_cache_dtype == "fp32":
#         key = key.to(torch.float32)
#         value = value.to(torch.float32)
#     else:
#         raise ValueError(f"不支持的kv_cache_dtype: {kv_cache_dtype}")

#     # ---------------- 3. 计算Flash Attention Cache的维度参数x ----------------
#     # x = 16 // 每个元素的字节数（INT8=1→x=16，FP16=2→x=8，FP32=4→x=4）
#     x = 16 // key.element_size()
#     num_blocks = key_cache.shape[0]

#     # ---------------- 4. 重塑K/V为Flash Attention Cache的维度 ----------------
#     # 展平batch和seq维度：[batch_size, num_heads, seq_len, head_dim] -> [batch_size*seq_len, num_heads, head_dim]
#     key_flat = key.reshape(-1, num_heads, head_dim)
#     value_flat = value.reshape(-1, num_heads, head_dim)

#     # 重塑为Flash Cache格式：
#     # Key Cache: [num_blocks, num_heads, head_dim//x, -1, x]
#     # Value Cache: [num_blocks, num_heads, head_dim, -1]
#     key_reshaped = key_flat.view(-1, num_heads, head_dim // x, 1, x)
#     value_reshaped = value_flat.view(-1, num_heads, head_dim, 1)

#     # ---------------- 5. 根据slot_mapping批量写入Cache ----------------
#     # 验证slot_mapping的有效性
#     valid_slots = slot_mapping[slot_mapping >= 0]
#     if len(valid_slots) == 0:
#         return  # 无有效槽位，直接返回

#     # 批量写入（替代for循环，提升效率）
#     # 只写入有效槽位
#     key_cache[valid_slots] = key_reshaped[slot_mapping >= 0]
#     value_cache[valid_slots] = value_reshaped[slot_mapping >= 0]