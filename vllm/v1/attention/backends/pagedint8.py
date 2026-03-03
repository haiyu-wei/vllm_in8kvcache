# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

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
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

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
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
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
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "INT8_PAGE_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Int8PageAttention supports all attention types."""
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
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
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
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    # ADD:
    @staticmethod
    def get_int8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype == "int8":
            return torch.int8
        else:
            raise ValueError(f"Unrecognized int8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        # ADD
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
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, INT8PAttnImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


# 把 vLLM 高层通用的注意力请求，转换为 FlashAttention 底层算子能看懂的特定数据结构，
# 同时处理 CUDA Graph 捕获、AOT 调度、Paged KV Cache 等高级特性
class INT8PAGEDATTENMetadataBuilder(AttentionMetadataBuilder[INT8PAGEDATTENMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
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

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> INT8PAGEDATTENMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
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
                    causal=causal,
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
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
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
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = INT8PAGEDATTENMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

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

    # 把这个禁用
    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


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
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        # 【INT8 新增】放宽检查，允许 INT8
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if self.kv_cache_dtype.startswith("fp8") and not flash_attn_supports_fp8():
                raise NotImplementedError(
                    "FlashAttention does not support fp8 kv-cache on this device."
                )
            # INT8 直接通过，假设硬件支持

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        # 【INT8 新增】INT8 量化相关初始化
        self.is_int8_kv = self.kv_cache_dtype.startswith("int8")
        if self.is_int8_kv:
            # 动态量化的缩放因子会在运行时更新
            self._int8_k_scale = None
            self._int8_v_scale = None

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
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for INT8PAttnImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens
        # 保存原始dtype（用于反量化恢复）
        target_dtype = query.dtype

        # -------------------------- Encoder Attention --------------------------
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # Encoder模式：Q/K/V都做INT8量化
            if self.is_int8_kv:
                # 计算Q/K/V缩放因子
                q_scale = self._calc_scale_int8(query[:num_actual_tokens])
                k_scale = self._calc_scale_int8(key[:num_actual_tokens])
                v_scale = self._calc_scale_int8(value[:num_actual_tokens])
                
                # 量化+反量化（模拟推理流程）
                query_quant = self._quantize_int8(query[:num_actual_tokens], q_scale)
                key_quant = self._quantize_int8(key[:num_actual_tokens], k_scale)
                value_quant = self._quantize_int8(value[:num_actual_tokens], v_scale)
                
                query_dequant = self._dequantize_int8(query_quant, q_scale, target_dtype)
                key_dequant = self._dequantize_int8(key_quant, k_scale, target_dtype)
                value_dequant = self._dequantize_int8(value_quant, v_scale, target_dtype)
            else:
                query_dequant = query[:num_actual_tokens]
                key_dequant = key[:num_actual_tokens]
                value_dequant = value[:num_actual_tokens]

            return self._forward_encoder_attention(
                query_dequant,
                key_dequant,
                value_dequant,
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # -------------------------- Decoder Attention --------------------------
        key_cache, value_cache = kv_cache.unbind(0)
        query_actual = query[:num_actual_tokens]
        key_actual = key[:num_actual_tokens]
        value_actual = value[:num_actual_tokens]
        output_actual = output[:num_actual_tokens]

        # INT8量化/反量化核心逻辑
        q_descale = k_descale = v_descale = None
        if self.is_int8_kv:
            # 1. KV Cache反量化
            key_cache_int8 = key_cache.view(torch.int8)
            value_cache_int8 = value_cache.view(torch.int8)
            
            k_scale = self._int8_k_scale
            v_scale = self._int8_v_scale
            
            if k_scale is not None and v_scale is not None:
                # 恢复缩放因子维度：[num_kv_heads] -> [1, num_kv_heads, 1]
                k_scale = k_scale.unsqueeze(0).unsqueeze(-1)
                v_scale = v_scale.unsqueeze(0).unsqueeze(-1)
                
                # 反量化并恢复原始dtype
                key_cache = self._dequantize_int8(key_cache_int8, k_scale, target_dtype)
                value_cache = self._dequantize_int8(value_cache_int8, v_scale, target_dtype)

            # 2. Query量化（和KV精度匹配）
            q_scale = self._calc_scale_int8(query_actual)
            query_quant = self._quantize_int8(query_actual, q_scale)
            query_actual = self._dequantize_int8(query_quant, q_scale, target_dtype)
            
            # 3. 当前step的K/V量化（可选，保持一致性）
            k_scale_curr = self._calc_scale_int8(key_actual)
            v_scale_curr = self._calc_scale_int8(value_actual)
            
            key_quant = self._quantize_int8(key_actual, k_scale_curr)
            value_quant = self._quantize_int8(value_actual, v_scale_curr)
            
            key_actual = self._dequantize_int8(key_quant, k_scale_curr, target_dtype)
            value_actual = self._dequantize_int8(value_quant, v_scale_curr, target_dtype)

            # 4. 准备descale参数（供FlashAttention使用）
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
            q_descale = q_scale.squeeze(0).squeeze(-1).expand(descale_shape)
            k_descale = k_scale.squeeze(0).squeeze(-1).expand(descale_shape)
            v_descale = v_scale.squeeze(0).squeeze(-1).expand(descale_shape)

        elif self.kv_cache_dtype.startswith("fp8"):
            dtype = Int8PageAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # ========== 关键修复：调用Decoder Attention核心计算 ==========
        if self.dcp_world_size > 1:
            # DCP模式：调用_forward_with_dcp
            self._forward_with_dcp(
                query_actual,
                key_actual,
                value_actual,
                key_cache,
                value_cache,
                output_actual,
                attn_metadata,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        else:
            # 非DCP模式：直接调用FlashAttention
            seqused_k = attn_metadata.seq_lens[:attn_metadata.query_start_loc.shape[0]-1]
            sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None
            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
            flash_attn_varlen_func(
                q=query_actual,
                k=key_cache,  # Paged Attention使用KV Cache中的K
                v=value_cache,  # Paged Attention使用KV Cache中的V
                out=output_actual,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                # 关键1：传入seqused_k（必须和block_table同时传）
                seqused_k=seqused_k,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                # 关键2：传入block_table（分页KV Cache的块表）
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                # 关键3：传入scheduler_metadata（FA3需要）
                scheduler_metadata=attn_metadata.scheduler_metadata,
            )

        return output

    def _calc_scale_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """计算INT8对称量化的缩放因子（按head维度）"""
        # tensor shape: [num_tokens, num_heads, head_size]
        # 按head维度计算abs_max，保留num_heads维度
        abs_max = tensor.abs().amax(dim=(0, 2), keepdim=True)  # [1, num_heads, 1]
        # 对称量化缩放因子 = abs_max / 127.0
        scale = abs_max / 127.0
        # 避免除零，设置最小阈值
        scale = scale.clamp_min(1e-6)
        return scale

    def _quantize_int8(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """INT8对称量化：[-127, 127]"""
        tensor_scaled = tensor / scale
        # 裁剪到[-127, 127]，四舍五入后转int8
        tensor_clipped = torch.clamp(tensor_scaled, -127, 127)
        tensor_quant = torch.round(tensor_clipped).to(torch.int8)
        return tensor_quant

    def _dequantize_int8(self, tensor_int8: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """INT8反量化，恢复到目标dtype"""
        tensor_dequant = tensor_int8.to(target_dtype) * scale
        return tensor_dequant.contiguous()  # 保证内存连续

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
            # 1. 修复缩放因子计算维度（按num_tokens和head_size求max）
            # key shape: [num_tokens, num_kv_heads, head_size]
            k_abs_max = key.abs().amax(dim=(0, 2), keepdim=False)  # [num_kv_heads]
            v_abs_max = value.abs().amax(dim=(0, 2), keepdim=False)  # [num_kv_heads]
            
            # 2. 对称量化缩放因子（用127而非128）
            k_scale = k_abs_max / 127.0
            v_scale = v_abs_max / 127.0
            
            # 3. 避免除零
            k_scale = k_scale.clamp_min(1e-6)
            v_scale = v_scale.clamp_min(1e-6)
            
            # 4. 保存缩放因子（供forward反量化使用）
            self._int8_k_scale = k_scale
            self._int8_v_scale = v_scale
            
            # 5. 修复量化范围：[-127, 127]
            k_scale_exp = k_scale.unsqueeze(0).unsqueeze(-1)  # [1, num_kv_heads, 1]
            v_scale_exp = v_scale.unsqueeze(0).unsqueeze(-1)
            
            key_int8 = torch.clamp(key / k_scale_exp, -127, 127).to(torch.int8)
            value_int8 = torch.clamp(value / v_scale_exp, -127, 127).to(torch.int8)
            
            # 6. 存储量化后的KV到Cache
            key_cache = key_cache.view(torch.int8)
            value_cache = value_cache.view(torch.int8)
            
            from vllm.v1.attention.backends.pagedint8_util import reshape_and_cache_flash_pytorch
            reshape_and_cache_flash_pytorch(
                key_int8,
                value_int8,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
            return

        # 原有FP8/默认逻辑
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
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
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
        # FA returns LSE in shape [ H, B ] but cp_lse_ag_out_rs wants [ B, H ]
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: INT8PAGEDATTENMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        # flash_attn_varlen_func(
        #     q=query,
        #     k=key,
        #     v=value,
        #     out=output,
        #     cu_seqlens_q=cu_seqlens_q,
        #     cu_seqlens_k=cu_seqlens_k,
        #     max_seqlen_q=max_seqlen_q,
        #     max_seqlen_k=max_seqlen_k,
        #     softmax_scale=self.scale,
        #     causal=False,  # Encoder attention is bidirectional
        #     alibi_slopes=self.alibi_slopes,
        #     window_size=sliding_window_size,
        #     softcap=self.logits_soft_cap,
        #     fa_version=self.vllm_flash_attn_version,
        #     q_descale=layer._q_scale.expand(descale_shape),
        #     k_descale=layer._k_scale.expand(descale_shape),
        #     v_descale=layer._v_scale.expand(descale_shape),
        #     num_splits=1 if self.batch_invariant_enabled else 0,
        # )
        from vllm.v1.attention.backends.pagedint8_util import flash_attn_varlen_func_pytorch
        flash_attn_varlen_func_pytorch(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output

