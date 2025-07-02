from typing import Dict, Any, Optional, List
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.chat_models import ChatOpenAI, ChatHuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch


# --------------------- 基础配置（支持多模型类型）---------------------
class LLMConfig:
    # 通用参数（所有模型共享）
    MODEL_TYPE: str = "local"  # 可选: "local", "openai", "aliyun"（第三方API）
    MAX_INPUT_TOKEN: int = 4096  # 最大输入token数（需与模型实际支持一致）
    MAX_GENERATION_TOKEN: int = 1024  # 最大生成token数
    TEMPERATURE: float = 0.7  # 创造力控制（0.0→确定性，1.0→随机性）
    TOP_P: float = 0.9  # nucleus sampling阈值
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # 设备选择
    
    # 本地模型专属配置（以Qwen-7B-Chat为例）
    LOCAL_MODEL_NAME: str = "qwen-7b-chat"  # 模型别名（用于日志区分）
    LOCAL_MODEL_PATH: str = "/data/models/qwen-7b-chat"  # 本地模型路径
    QUANTIZATION_BITS: int = 4  # 量化位数（4/8位，0表示不量化）
    USE_FP16: bool = False  # 是否使用FP16（与量化互斥）
    N_THREADS: int = 8  # CPU推理线程数（仅CPU模式有效）
    
    # 第三方API配置（以OpenAI为例）
    OPENAI_API_KEY: str = "sk-xxx"  # 生产环境建议从环境变量读取
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-3.5-turbo-16k"  # 选择支持长上下文的型号
    
    # 中医问诊场景专属参数
    TCM_PROMPT_TEMPLATE: str = """
    你是一位中医执业医师，用户将描述症状，请根据中医理论进行诊断：
    1. 先分析病因病机（八纲辨证、脏腑辨证）
    2. 给出治法和推荐方剂（注明出处）
    3. 注意使用专业术语但避免生僻字
    【用户症状】：{symptom}
    【参考知识】：{retrieved_knowledge} （来自中医古籍知识库的检索结果）
    """


# --------------------- 模型加载工厂函数（支持动态切换）---------------------
def load_llm(config: LLMConfig) -> Any:
    """根据配置加载LLM模型（本地/第三方API自适应）"""
    if config.MODEL_TYPE == "local":
        return _load_local_model(config)
    elif config.MODEL_TYPE == "openai":
        return _load_openai_model(config)
    else:
        raise ValueError(f"不支持的模型类型：{config.MODEL_TYPE}")


def _load_local_model(config: LLMConfig) -> HuggingFacePipeline:
    """加载本地量化大模型（支持LLaMA2、Qwen、ChatGLM等）"""
    # 量化配置（4位量化节省75%显存，支持GPU推理）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.QUANTIZATION_BITS == 4,
        load_in_8bit=config.QUANTIZATION_BITS == 8,
        llm_int8_threshold=6.0,  # 混合精度阈值（8位量化专用）
        llm_int8_has_fp16_weight=False,
        bnb_4bit_use_double_quant=True,  # 双重量化进一步压缩
        bnb_4bit_quant_type="nf4",  # 正常浮点4位（优于fp4）
        bnb_4bit_compute_dtype=torch.float16 if config.USE_FP16 else torch.float32
    ) if config.QUANTIZATION_BITS > 0 else None

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.LOCAL_MODEL_PATH, 
        trust_remote_code=True, 
        use_fast=False  # 部分国产模型需关闭fast tokenizer
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_MODEL_PATH,
        quantization_config=quantization_config,
        device_map=config.DEVICE,  # 自动分配GPU/CPU资源
        trust_remote_code=True
    )
    
    # 创建推理管道（支持流式输出）
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=config.MAX_INPUT_TOKEN + config.MAX_GENERATION_TOKEN,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        device=0 if config.DEVICE.startswith("cuda") else -1  # GPU设备ID
    )
    return HuggingFacePipeline(pipeline=pipe)


def _load_openai_model(config: LLMConfig) -> ChatOpenAI:
    """加载OpenAI API模型（带速率限制和重试机制）"""
    return ChatOpenAI(
        model_name=config.OPENAI_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_GENERATION_TOKEN,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_API_BASE,
        request_timeout=60,  # 长响应超时控制（单位：秒）
        max_retries=3,  # 网络波动自动重试
        streaming=True  # 启用流式响应（提升用户体验）
    )


# --------------------- 领域专用链初始化（以中医问诊为例）---------------------
def init_tcm_llm_chain(config: LLMConfig) -> LLMChain:
    """创建中医问诊专用LLM Chain（结合知识库检索结果）"""
    # 初始化LLM
    llm = load_llm(config)
    
    # 定义领域专用提示模板（包含检索结果注入点）
    prompt = PromptTemplate(
        input_variables=["symptom", "retrieved_knowledge"],
        template=config.TCM_PROMPT_TEMPLATE
    )
    
    return LLMChain(llm=llm, prompt=prompt)


# --------------------- 性能优化配置（生产环境关键参数）---------------------
class PerformanceConfig:
    # 缓存配置（减少重复计算）
    CACHE_ENABLED: bool = True
    CACHE_TYPE: str = "sqlite"  # 可选: "in_memory", "redis"（分布式缓存）
    CACHE_TTL: int = 3600  # 缓存有效期（秒）
    
    # 并行推理配置（多GPU/多实例）
    PARALLEL_GPU_COUNT: int = 1  # 单卡部署设为1，多卡需配合FSDP
    THREAD_POOL_SIZE: int = 10  # 异步请求线程池大小（API服务专用）
    
    # 显存优化策略
    OFFLOAD_CPU: bool = False  # 是否将非激活层卸载到CPU（适合显存不足）
    USE_TRITON: bool = True  # 启用Triton优化内核（提升20%推理速度）


# --------------------- 安全策略配置（防止恶意输入）---------------------
class SecurityConfig:
    # 输入内容过滤
    PROHIBITED_KEYWORDS: List[str] = [
        "恶意代码", "攻击指令", "敏感政治", "虚假信息"
    ]  # 自定义敏感词库（建议从文件加载）
    MAX_INPUT_LENGTH: int = 8192  # 输入内容字符数限制（防止prompt注入）
    MAX_OUTPUT_LENGTH: int = 4096  # 输出内容字符数限制
    
    # 权限控制（多租户场景）
    TENANT_TOKEN_BLACKLIST: List[str] = []  # 禁用的租户令牌
    REQUEST_RATE_LIMIT: Dict[str, str] = {  # 速率限制（IP: "5/分钟"）
        "192.168.1.1": "10/分钟",
        "*": "5/分钟"  # 全局默认限制
    }
