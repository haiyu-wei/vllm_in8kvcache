# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

# If FA is available, we still use it for non-int8 / encoder / dcp paths where appropriate.
if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )

from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# -------------------------
# Backend + metadata builder
# -------------------------

class Int8PageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.int8]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "INT8_PAGE_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["INT8PAttnImpl"]:
        return INT8PAttnImpl

    @staticmethod
    def get_builder_cls() -> type["INT8PAGEDATTENMetadataBuilder"]:
        return INT8PAGEDATTENMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        if kv_cache_dtype == "int8":
            return True
        return kv_cache_dtype in ["auto", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class INT8PAGEDATTENMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(vllm_config: VllmConfig) -> set[tuple[int, int] | None]:
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, INT8PAttnImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class INT8PAGEDATTENMetadataBuilder(AttentionMetadataBuilder[INT8PAGEDATTENMetadata]):
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = self.parallel_config.cp_kv_cache_interleave_size
        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            self.max_num_splits = self.attention_config.flash_attn_max_num_splits_for_cuda_graph

        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> INT8PAGEDATTENMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sw = sliding_window_configs.pop()
                    if sw is not None:
                        self.aot_sliding_window = sw
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal_flag):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = Int8PageAttentionBackend.get_fp8_dtype_for_flashattn(cache_dtype)
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal_flag,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens
            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal_flag=False,
            )
        elif use_cascade:
            # keep cascade disabled in impl; we still build metadata safely
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal_flag=causal,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal_flag=causal,
            )

        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        return INT8PAGEDATTENMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=False,                 # force disabled
            common_prefix_len=0,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )

    def update_block_table(
        self,
        metadata: INT8PAGEDATTENMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> INT8PAGEDATTENMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


# -------------------------
# 主要实现
# -------------------------

class INT8PAttnImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        # 禁用
        self.alibi_slopes = None

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        # DCP info (set in builder; keep defaults here)
        try:
            self.dcp_world_size = get_dcp_group().world_size
        except Exception:
            self.dcp_world_size = 1

        # allow quantized kv cache
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if self.kv_cache_dtype.startswith("fp8") and not flash_attn_supports_fp8():
                raise NotImplementedError("FlashAttention does not support fp8 kv-cache on this device.")
            # int8 accepted

        self.sinks = sinks

        self.is_int8_kv = self.kv_cache_dtype == "int8"
        self._int8_k_scale: Optional[torch.Tensor] = None  # [Hkv]
        self._int8_v_scale: Optional[torch.Tensor] = None  # [Hkv]

        self.supports_quant_query_input = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: INT8PAGEDATTENMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for INT8PAttnImpl")

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        # ---------------- Encoder ----------------
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # Keep your encoder logic simple: no kv-cache paged mode
            # If you need int8 for encoder, you should do QKV quant separately (not covered here).
            cu_seqlens_q = attn_metadata.query_start_loc
            cu_seqlens_k = attn_metadata.query_start_loc
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_query_len

            sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

            # Use FA if available; else fall back to simple pytorch (non-paged)
            if is_flash_attn_varlen_func_available():
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key[:num_actual_tokens],
                    v=value[:num_actual_tokens],
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=False,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    softcap=self.logits_soft_cap,
                    fa_version=self.vllm_flash_attn_version,
                    num_splits=1 if self.batch_invariant_enabled else 0,
                )
                return output

            # minimal fallback encoder (non-paged) not implemented in this file
            output[:num_actual_tokens].zero_()
            return output

        # ---------------- Decoder (Paged) ----------------
        key_cache, value_cache = kv_cache.unbind(0)

        query_actual = query[:num_actual_tokens]
        output_actual = output[:num_actual_tokens]

        q_descale = None
        k_descale = None
        v_descale = None

        # IMPORTANT: Do NOT dequantize whole cache here.
        if self.is_int8_kv:
            key_cache = key_cache.view(torch.int8)
            value_cache = value_cache.view(torch.int8)

            assert self._int8_k_scale is not None and self._int8_v_scale is not None, (
                "INT8 kv-cache scales not initialized. do_kv_cache_update must run before forward."
            )

            
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads) # [batch, Hkv]
            k_descale = self._int8_k_scale.view(1, -1).expand(descale_shape).contiguous()
            v_descale = self._int8_v_scale.view(1, -1).expand(descale_shape).contiguous()

        elif self.kv_cache_dtype.startswith("fp8"):
            dtype = Int8PageAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # DCP path keeps FA (unchanged)
        if getattr(self, "dcp_world_size", 1) > 1:
            return self._forward_with_dcp(
                query_actual,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                output_actual,
                attn_metadata,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        # use paged attention
        seqused_k = attn_metadata.seq_lens[: attn_metadata.query_start_loc.shape[0] - 1]
        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        # 原本的算子调用
        if is_flash_attn_varlen_func_available() and not self.is_int8_kv:
            flash_attn_varlen_func(
                q=query_actual,
                k=key_cache,
                v=value_cache,
                out=output_actual,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=seqused_k,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
                num_splits=attn_metadata.max_num_splits,
                scheduler_metadata=attn_metadata.scheduler_metadata,
            )
            return output

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

        # DEBUG
        print(f"DTYPE IN FORWARD: query {query.dtype}, key_cache {key_cache.dtype}, value_cache {value_cache.dtype}, output {output.dtype}")
        print(f"SHAPE IN FORWARD: query {query.shape}, key_cache {key_cache.shape}, value_cache {value_cache.shape}, output {output.shape}")
        return output

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

    # 数据上下文并行data context parallel 这里先不用
    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: INT8PAGEDATTENMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert is_flash_attn_varlen_func_available(), "DCP path requires FlashAttention."

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )

        # merge states (keep your original merge logic if you use it elsewhere)
        # Here we simply write context_attn_out back if needed (placeholder).
        output.copy_(context_attn_out)
        return output