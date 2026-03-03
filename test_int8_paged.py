import torch
import torch.nn as nn
import numpy as np
from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple

# ====================== 1. 固定随机性 ======================
def set_seed(seed: int = 42):
    """固定所有随机种子（保证可复现）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

# ====================== 2. 核心配置（适配vLLM限制） ======================
MODEL_PATH = "Qwen/Qwen-7B-Chat"
MAX_MODEL_LEN = 512
GPU_MEM_UTIL = 0.8
TENSOR_PARALLEL_SIZE = 1
MAX_LOGPROBS = 20  # vLLM强制限制：最大只能取20

# 固定输入（简洁易验证）
TEST_PROMPT = "<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n"

# 采样参数：logprobs改为20（适配vLLM限制）
sampling_params = SamplingParams(
    temperature=0.0,          # 贪心解码，无随机
    max_tokens=32,            # 生成32个Token
    top_p=1.0,
    stop=["<|im_end|>"],
    seed=42,
    logprobs=MAX_LOGPROBS,    # 改为vLLM允许的最大值20
    prompt_logprobs=MAX_LOGPROBS
)

# ====================== 3. Logits 比对工具（适配Top-20） ======================
def logits_metrics(logits_native: torch.Tensor, logits_int8: torch.Tensor, name: str = "logits") -> Dict:
    """
    计算Logits的核心精度指标（适配vLLM的Top-20限制）
    :param logits_native: 原生Backend的Logits [seq_len, 20]
    :param logits_int8: INT8 Backend的Logits [seq_len, 20]
    :return: 多维度精度指标
    """
    # 对齐序列长度（取较短的）
    min_seq_len = min(logits_native.shape[0], logits_int8.shape[0])
    logits_native = logits_native[:min_seq_len].float().cpu()
    logits_int8 = logits_int8[:min_seq_len].float().cpu()

    # 空张量保护
    if logits_native.numel() == 0 or logits_int8.numel() == 0:
        return {
            "name": name,
            "avg_cos_sim": 0.0,
            "mae": 1e9,
            "mre": 1e9,
            "top1_match_rate": 0.0,
            "avg_top20_match": 0.0,
            "is_high_precision": False
        }

    # 1. 余弦相似度（衡量分布相似性，越接近1越好）
    cos_sim_list = []
    for i in range(min_seq_len):
        sim = torch.nn.functional.cosine_similarity(
            logits_native[i].flatten(), 
            logits_int8[i].flatten(), 
            dim=0
        ).item()
        cos_sim_list.append(sim)
    avg_cos_sim = np.mean(cos_sim_list)

    # 2. 平均绝对误差（MAE）
    mae = torch.abs(logits_native - logits_int8).mean().item()

    # 3. 平均相对误差（MRE）
    mre = (torch.abs(logits_native - logits_int8) / (torch.abs(logits_native) + 1e-8)).mean().item()

    # 4. Top-1 Token命中率（原Token ID对比）
    top1_native = torch.argmax(logits_native, dim=-1)
    top1_int8 = torch.argmax(logits_int8, dim=-1)
    top1_match = (top1_native == top1_int8).float().mean().item()

    # 5. Top-20 Token匹配率（适配vLLM限制，替代原Top-100）
    top20_match = []
    for i in range(min_seq_len):
        native_top20 = set(torch.topk(logits_native[i], k=MAX_LOGPROBS).indices.numpy())
        int8_top20 = set(torch.topk(logits_int8[i], k=MAX_LOGPROBS).indices.numpy())
        match_rate = len(native_top20 & int8_top20) / MAX_LOGPROBS
        top20_match.append(match_rate)
    avg_top20_match = np.mean(top20_match)

    return {
        "name": name,
        "avg_cos_sim": avg_cos_sim,       # 核心指标：分布相似性
        "mae": mae,                       # 数值误差
        "mre": mre,                       # 相对误差
        "top1_match_rate": top1_match,    # Token ID命中率
        "avg_top20_match": avg_top20_match,  # 适配vLLM的Top-20匹配率
        "is_high_precision": avg_cos_sim >= 0.99 and mae <= 1e-3  # 高精度阈值
    }

def print_metrics(metrics: Dict):
    """美观打印Logits比对结果（适配Top-20）"""
    print(f"\n{'='*80}")
    print(f"📊 {metrics['name']} 精度指标")
    print(f"{'='*80}")
    print(f"平均余弦相似度 (越接近1越好): {metrics['avg_cos_sim']:.6f}")
    print(f"平均绝对误差 (越小越好): {metrics['mae']:.6f}")
    print(f"平均相对误差 (越小越好): {metrics['mre']:.6f}")
    print(f"Top-1 Token匹配率 (原ID对比): {metrics['top1_match_rate']:.6f}")
    print(f"Top-20 Token匹配率 (适配vLLM): {metrics['avg_top20_match']:.6f}")
    print(f"是否达到高精度标准: {'✅' if metrics['is_high_precision'] else '❌'}")

# ====================== 4. 从vLLM输出中提取Logits（修复Logprob对象问题） ======================
def extract_logits_from_output(output) -> torch.Tensor:
    """
    从vLLM的generate输出中还原Logits张量（修复Logprob对象类型错误）
    :param output: vLLM generate返回的单个输出对象
    :return: Logits张量 [seq_len, 20]
    """
    logits_list = []
    
    # 辅助函数：提取Logprob对象的数值
    def get_logprob_value(logprob_obj):
        """将vLLM的Logprob对象转换为浮点数值"""
        # 兼容不同vLLM版本的Logprob对象
        if hasattr(logprob_obj, 'logprob'):
            return float(logprob_obj.logprob)  # 取对数概率（核心值）
        elif hasattr(logprob_obj, 'prob'):
            return float(logprob_obj.prob)    # 备选：取普通概率
        else:
            return float(logprob_obj)         # 兜底：直接转浮点
    
    # 1. 提取Prompt部分的LogProb（修复Logprob对象）
    if output.prompt_logprobs:
        for prompt_logprob in output.prompt_logprobs:
            if prompt_logprob is None:
                continue
            # 提取Top-20 Token的LogProb并转换为数值
            token_items = list(prompt_logprob.items())[:MAX_LOGPROBS]  # 限制为20个
            # 按Token ID排序，保证不同Backend的顺序一致
            token_items.sort(key=lambda x: x[0])
            # 提取并转换Logprob数值
            logprobs = [get_logprob_value(item[1]) for item in token_items]
            # 构造张量（补0到20维）
            logits = torch.tensor(logprobs, dtype=torch.float32)
            if len(logits) < MAX_LOGPROBS:
                logits = torch.cat([logits, torch.zeros(MAX_LOGPROBS - len(logits))])
            logits_list.append(logits)
    
    # 2. 提取生成部分的LogProb（修复Logprob对象）
    if output.outputs[0].logprobs:
        for token_logprob in output.outputs[0].logprobs:
            if token_logprob is None:
                continue
            # 提取Top-20 Token的LogProb并转换为数值
            token_items = list(token_logprob.items())[:MAX_LOGPROBS]
            token_items.sort(key=lambda x: x[0])  # 按Token ID排序
            logprobs = [get_logprob_value(item[1]) for item in token_items]
            # 构造张量（补0到20维）
            logits = torch.tensor(logprobs, dtype=torch.float32)
            if len(logits) < MAX_LOGPROBS:
                logits = torch.cat([logits, torch.zeros(MAX_LOGPROBS - len(logits))])
            logits_list.append(logits)
    
    # 3. 转换为张量
    if logits_list:
        return torch.stack(logits_list)
    else:
        return torch.zeros((0, MAX_LOGPROBS))

# ====================== 5. 运行单个Backend并提取Logits ======================
def run_backend(
    backend_name: str,
    attention_backend: str,
    kv_cache_dtype: str,
    prompt: str,
    sampling_params: SamplingParams
) -> Dict:
    """运行单个Backend，返回Logits和生成结果"""
    print(f"\n🚀 启动 Backend: {backend_name} (KV Cache: {kv_cache_dtype})")

    # 初始化LLM（强制Eager模式，保证可复现）
    llm = LLM(
        model=MODEL_PATH,
        attention_backend=attention_backend,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=True,  # 禁用JIT，避免随机因素
    )

    # 预热（避免首次运行的初始化干扰）
    llm.generate(prompt, sampling_params)
    set_seed(42)  # 重置种子

    # 正式推理
    outputs = llm.generate(prompt, sampling_params)
    output = outputs[0]

    # 提取Logits（修复Logprob对象后）
    logits = extract_logits_from_output(output)

    return {
        "backend_name": backend_name,
        "logits": logits,                  # 核心比对对象：Logits
        "token_ids": torch.tensor(output.outputs[0].token_ids),
        "output_text": output.outputs[0].text
    }

# ====================== 6. 主比对逻辑 ======================
if __name__ == "__main__":
    # 1. 运行原生Backend（FP16）
    native_data = run_backend(
        backend_name="原生Backend (FP16)",
        attention_backend="FLASH_ATTN",
        kv_cache_dtype="auto",
        prompt=TEST_PROMPT,
        sampling_params=sampling_params
    )

    # 2. 运行INT8 Backend
    int8_data = run_backend(
        backend_name="INT8 Backend",
        attention_backend="INT8_PAGE_ATTN",
        kv_cache_dtype="int8",
        prompt=TEST_PROMPT,
        sampling_params=sampling_params
    )

    # 3. 核心：比对Logits（适配Top-20）
    logits_metrics_result = logits_metrics(
        native_data["logits"],
        int8_data["logits"],
        name="端到端Logits对比（Top-20）"
    )
    print_metrics(logits_metrics_result)

    # 4. 打印生成文本（人工验证）
    print(f"\n{'='*80}")
    print(f"📝 生成文本对比")
    print(f"{'='*80}")
    print(f"原生Backend: {native_data['output_text']}")
    print(f"INT8 Backend: {int8_data['output_text']}")