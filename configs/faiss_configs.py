import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import Optional, Dict, Any

# --------------------- 基础配置（全局参数）---------------------
class FaissGPUConfig:
    DEVICE_ID = 0  # 单GPU设备ID（多卡需配合data parallel）
    EMBEDDING_DIM = 1024  # 向量维度（需与Embedding模型输出一致，如BGE-large为1024）
    INDEX_TYPE = "GPU_FLAT"  # 可选: GPU_FLAT, GPU_IVF, GPU_IVF_PQ, CPU_FLAT（调试用）
    USE_FP16 = True  # 使用FP16混合精度（节省50%显存，不影响检索精度）
    IVF_NLIST = 1024  # IVF索引聚类中心数（数据量>10万时建议设为2^10-2^14）
    PQ_M = 8  # PQ量化参数（仅INDEX_TYPE包含PQ时有效，建议4-16）

# --------------------- GPU资源初始化（核心函数）---------------------
def get_gpu_resources() -> faiss.StandardGpuResources:
    """初始化Faiss GPU资源（支持显存复用优化）"""
    res = faiss.StandardGpuResources()
    # 配置GPU选项（关键：允许显存增长，避免占用全部显存）
    cfg = faiss.GpuOptions()
    cfg.allow_gpu_memory_growth = True  # 按需申请显存（适合多任务环境）
    res.set_gpu_options(cfg)
    return res

# --------------------- 索引创建工厂函数（支持多种索引类型）---------------------
def create_faiss_index(
    embedding_model: Embeddings,
    config: FaissGPUConfig = FaissGPUConfig()
) -> FAISS:
    """
    创建Faiss索引（CPU/GPU版本自动切换）
    :param embedding_model: LangChain Embedding对象（需提前获取维度）
    :param config: 配置参数对象
    :return: FAISS向量存储对象
    """
    # 检查维度一致性
    if config.EMBEDDING_DIM != embedding_model.dimension:
        raise ValueError(
            f"索引维度{config.EMBEDDING_DIM}与Embedding输出维度{embedding_model.dimension}不匹配"
        )
    
    if config.INDEX_TYPE.startswith("GPU"):
        return _create_gpu_index(embedding_model, config)
    else:
        return _create_cpu_index(embedding_model, config)

# --------------------- GPU索引创建（核心实现）---------------------
def _create_gpu_index(
    embedding_model: Embeddings,
    config: FaissGPUConfig
) -> FAISS:
    """创建GPU加速的Faiss索引（支持Flat/IVF/PQ等类型）"""
    res = get_gpu_resources()  # 初始化GPU资源
    faiss_index: faiss.Index
    
    if config.INDEX_TYPE == "GPU_FLAT":
        # 最精确的索引（适合数据量<100万，追求精度场景）
        index_config = faiss.GpuIndexFlatConfig()
        index_config.device = config.DEVICE_ID
        index_config.use_float16 = config.USE_FP16  # 启用FP16
        faiss_index = faiss.GpuIndexFlatL2(res, config.EMBEDDING_DIM, index_config)
    
    elif config.INDEX_TYPE == "GPU_IVF":
        # 倒排索引（适合百万级数据，速度提升5-10倍，精度轻微下降）
        ivf_config = faiss.GpuIndexIVFFlatConfig()
        ivf_config.device = config.DEVICE_ID
        ivf_config.use_float16 = config.USE_FP16
        # 必须先训练索引（使用部分数据生成聚类中心）
        faiss_index = faiss.GpuIndexIVFFlat(
            res, 
            config.EMBEDDING_DIM, 
            config.IVF_NLIST, 
            faiss.METRIC_L2, 
            False,  # 不自动训练（需手动调用index.train()）
            ivf_config
        )
    
    elif config.INDEX_TYPE == "GPU_IVF_PQ":
        # 乘积量化索引（适合千万级数据，显存占用降低75%）
        pq_config = faiss.GpuIndexIVFPQRConfig()
        pq_config.device = config.DEVICE_ID
        pq_config.use_float16 = config.USE_FP16
        faiss_index = faiss.GpuIndexIVFPQR(
            res, 
            config.EMBEDDING_DIM, 
            config.IVF_NLIST, 
            config.PQ_M,  # PQ段数（M=8表示每个向量分为8段量化）
            8,  # 每个段的量化字节数（固定8，即256种状态）
            faiss.METRIC_L2, 
            pq_config
        )
    
    else:
        raise ValueError(f"不支持的GPU索引类型：{config.INDEX_TYPE}")
    
    # 转换为LangChain可用的FAISS对象（注意：初始索引为空）
    return FAISS(
        index=faiss_index,
        embedding_function=embedding_model.embed_query,
        metadatas=[],  # 元数据后续添加
        index_to_docstore=None,
        docstore=None
    )

# --------------------- CPU索引创建（调试/小数据量用）---------------------
def _create_cpu_index(
    embedding_model: Embeddings,
    config: FaissGPUConfig
) -> FAISS:
    """创建CPU版本Faiss索引（用于开发调试或无GPU环境）"""
    if config.INDEX_TYPE == "CPU_FLAT":
        faiss_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    else:
        raise NotImplementedError("CPU仅支持FLAT索引")
    return FAISS.from_embeddings(
        embedding=embedding_model,
        index=faiss_index
    )

# --------------------- 索引持久化工具函数---------------------
def save_faiss_index(vectorstore: FAISS, path: str) -> None:
    """保存Faiss索引到磁盘（支持GPU索引直接序列化）"""
    faiss.write_index(vectorstore.index, path)

def load_faiss_index(
    embedding_model: Embeddings,
    path: str,
    config: FaissGPUConfig = FaissGPUConfig()
) -> FAISS:
    """从磁盘加载Faiss索引（自动识别GPU/CPU版本）"""
    faiss_index = faiss.read_index(path)
    # 关键：将CPU索引转换为GPU索引（如果配置为GPU模式）
    if config.INDEX_TYPE.startswith("GPU") and not faiss_index.is_gpu:
        res = get_gpu_resources()
        faiss_index = faiss.index_cpu_to_gpu(res, config.DEVICE_ID, faiss_index)
    return FAISS(
        index=faiss_index,
        embedding_function=embedding_model.embed_query,
        metadatas=[],
        index_to_docstore=None,
        docstore=None
    )
