from vllm import LLM, SamplingParams

# ====================== 1. 核心配置（含FP8 KV Cache） ======================
# 初始化vLLM引擎，关键配置FP8 KV Cache
llm = LLM(
    model="Qwen/Qwen-7B-Chat",
    attention_backend="INT8_PAGE_ATTN",
    kv_cache_dtype="int8",  # 先用 FP8 测试
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    trust_remote_code=True,
    enforce_eager=True,  # disable cuda graph
)

# ====================== 2. 生成参数配置 ======================
# 采样参数：控制生成效果，和API服务参数一致
sampling_params = SamplingParams(
    temperature=0.7,  # 随机性，0为确定性输出
    max_tokens=2000,   # 最大生成token数
    top_p=0.8,        # 核采样
    stop=["<|im_end|>"]  # Qwen的停止符，避免生成多余内容
)

# ====================== 3. 构造Qwen7B-Chat的对话格式 ======================
# Qwen需要固定的对话模板（<|im_start|>/<|im_end|>标签）
def build_qwen_prompt(user_input):
    """构造Qwen7B-Chat要求的prompt格式"""
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# 测试用的用户问题列表（可替换为自己的问题）
user_questions = [
    "你好，请介绍一下自己",
    "用Python写一个快速排序的示例代码"
]

# 构造符合Qwen格式的prompt列表
prompts = [build_qwen_prompt(question) for question in user_questions]

# ====================== 4. 本地推理（无服务，直接生成） ======================
# 执行生成（批量/单条都支持）
outputs = llm.generate(prompts, sampling_params)

# ====================== 5. 解析并打印结果 ======================
print("===== 推理结果 =====")
for output in outputs:
    # 提取原始prompt（用户问题）
    prompt = output.prompt
    # 提取生成的回复（过滤掉prompt部分）
    generated_text = output.outputs[0].text.strip()
    # 提取用户问题（从prompt中解析，可选）
    user_question = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    
    # 格式化输出
    print(f"\n用户问题：{user_question}")
    print(f"模型回复：{generated_text}")
    print("-" * 80)