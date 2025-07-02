import os
import faiss
import json  # 添加缺失的导入
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# --------------------- 配置参数 ---------------------
# 向量数据库配置
USE_GPU = True  # 是否使用GPU加速
NEAREST_NEIGHBORS = 5  # 检索最近邻数量
SCORE_THRESHOLD = 0.6  # 相似度阈值

# 本地模型路径配置
LOCAL_EMBEDDING_MODEL_PATH = "/root/autodl-tmp/medical/models/text2vec-large-chinese"  # 嵌入模型本地路径
LOCAL_LLM_MODEL_PATH = "/root/autodl-tmp/medical/models/Qwen2.5-7B"  # 大模型本地路径

# --------------------- 路径配置 ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件路径
DATA_DIR = os.path.join(BASE_DIR, "data")  # 中医古籍数据目录
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")  # 索引存储目录
FAISS_INDEX_NAME = "medical_knowledge"  # 索引名称

# JSON缓存文件路径
JSON_CACHE_FILE = os.path.join(DATA_DIR, "all_chunks.cache.json")

# 确保目录存在
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# --------------------- 数据加载 ---------------------
def load_segmented_documents() -> List[Document]:
    """从JSON缓存文件加载分割后的文档"""
    docs = []
    
    # 检查缓存文件是否存在
    if not os.path.exists(JSON_CACHE_FILE):
        print(f"警告：未找到缓存文件 {JSON_CACHE_FILE}")
        return docs
    
    print(f"从缓存加载文档: {JSON_CACHE_FILE}")
    
    try:
        with open(JSON_CACHE_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        print(f"找到 {len(chunks)} 个文档块")
        
        # 转换为LangChain Document对象
        for chunk in chunks:
            # 确保元数据是字典类型
            metadata = chunk.get('元数据', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # 创建Document对象
            doc = Document(
                page_content=chunk.get('内容', ''),
                metadata=metadata
            )
            docs.append(doc)
        
        print(f"成功加载 {len(docs)} 个文档")
        return docs
    except Exception as e:
        print(f"加载缓存文件失败: {str(e)}")
        return []

# --------------------- 向量化与FAISS索引构建 ---------------------
def build_vector_store() -> FAISS:
    """构建FAISS向量库"""
    # 1. 初始化向量化模型
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    
    print(f"正在加载本地嵌入模型: {LOCAL_EMBEDDING_MODEL_PATH}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device}
    )
    
    # 2. 检查索引文件是否存在 - 修复扩展名问题
    faiss_index_path = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.faiss")
    pkl_index_path = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.pkl")
    
    # 检查两个索引文件是否都存在
    if os.path.exists(faiss_index_path) and os.path.exists(pkl_index_path):
        # 加载现有索引
        print(f"加载现有FAISS索引: {faiss_index_path} 和 {pkl_index_path}")
        vector_store = FAISS.load_local(
            folder_path=FAISS_INDEX_DIR,
            embeddings=embedding_model,
            index_name=FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True  # 关键修复
        )
    else:
        # 创建新索引
        docs = load_segmented_documents()
        if not docs:
            raise ValueError(f"未找到任何文档，请检查 {JSON_CACHE_FILE}")
        
        print(f"创建新FAISS索引，文档数量: {len(docs)}")
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(
            folder_path=FAISS_INDEX_DIR,
            index_name=FAISS_INDEX_NAME
        )
        print(f"索引已保存至: {FAISS_INDEX_DIR}，索引名称: {FAISS_INDEX_NAME}")
    
    return vector_store

# --------------------- 问答系统初始化 ---------------------
def initialize_qa_chain() -> RetrievalQA:
    """初始化问答链"""
    # 1. 加载向量数据库
    vector_store = build_vector_store()
    
    # 2. 加载大模型 - 使用本地模型
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    print(f"正在加载本地大模型: {LOCAL_LLM_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_LLM_MODEL_PATH, 
        trust_remote_code=True,
        use_fast=False  # 避免一些兼容性问题
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # 创建HuggingFace管道
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,  # 减少生成长度节省显存
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. 构建上下文压缩检索器
    print("构建上下文压缩检索器...")
    compressor = LLMChainFilter.from_llm(
        llm=llm,
        prompt=PromptTemplate.from_template(
            "判断以下中医古籍内容是否与问题相关:\n问题: {query}\n内容: {text}\n相关吗? (是/否):"
        )
    )
    
    retriever = ContextualCompressionRetriever(
        base_retriever=vector_store.as_retriever(
            search_kwargs={
                "k": NEAREST_NEIGHBORS,
                "score_threshold": SCORE_THRESHOLD
            }
        ),
        base_compressor=compressor
    )
    
    # 4. 定义中医问答专用提示模板

    prompt_template = """
    作为中医古籍智能问答系统，请基于以下古籍原文回答问题:
    
    【古籍原文】：
    {context}
    
    【用户问题】：
    {question}
    
    要求:
    1. 回答必须基于提供的古籍原文
    2. 回答要简洁准确
    3. 涉及药方需明确药材剂量
    4. 使用专业术语但避免生僻字
    5. 如果原文没有直接相关信息，可以结合中医理论合理推断
    """
    prompt = PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# --------------------- 终端交互 ---------------------
def interactive_console(qa_chain: RetrievalQA):
    """命令行交互界面"""
    print("\n===== 中医古籍智能问答系统 =====")
    print("输入格式：直接提问（支持中医病症、方剂、药材等问题）")
    print("输入'quit'退出，输入'source'查看参考原文\n")
    
    while True:
        query = input("用户提问 >> ").strip()
        if query.lower() == "quit":
            break
        
        try:
            print("\n正在检索古籍数据库...")
            result = qa_chain({"query": query})
            
            print("\nAI解答 >>")
            print(result["result"])
            
            if "source_documents" in result and len(result["source_documents"]) > 0:
                print("\n参考古籍 >>")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"{i+1}. 来源：{doc.metadata.get('source', '未知')}")
                    if query.lower() == "source":
                        print(f"   原文片段：{doc.page_content[:200]}...")
            
            print("\n" + "="*50 + "\n")
        
        except faiss.Error as e:
            print(f"检索引擎错误：{str(e)}")
        except torch.cuda.OutOfMemoryError:
            print("显存不足！请尝试减少检索数量（修改NEAREST_NEIGHBORS参数）")
        except Exception as e:
            print(f"系统异常：{str(e)}")

# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    # 1. 环境检测
    if USE_GPU and not torch.cuda.is_available():
        print("警告：检测到GPU配置但无可用CUDA设备，将使用CPU模式")
        USE_GPU = False
    
    # 2. 检查数据目录
    if not os.path.exists(JSON_CACHE_FILE):
        print(f"警告：未找到缓存文件 {JSON_CACHE_FILE}")
        print("请确保已运行数据处理脚本并生成缓存文件")
    
    # 3. 加载问答系统
    try:
        print("正在初始化中医古籍问答系统...")
        qa_chain = initialize_qa_chain()
        print("系统初始化成功！")
    except Exception as e:
        print(f"系统初始化失败：{str(e)}")
        print(f"请检查：")
        print(f"1. 嵌入模型是否在 {LOCAL_EMBEDDING_MODEL_PATH}")
        print(f"2. 大模型是否在 {LOCAL_LLM_MODEL_PATH}")
        print(f"3. 缓存文件是否在 {JSON_CACHE_FILE}")
        exit(1)
    
    # 4. 启动交互界面
    interactive_console(qa_chain)
