import os
import faiss
import json
import torch
import numpy as np
from typing import List, Optional, Dict
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import warnings

# 忽略LangChain弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")

# --------------------- 配置参数 ---------------------
# 向量数据库配置
USE_GPU = True  # 是否使用GPU加速
NEAREST_NEIGHBORS = 5  # 检索最近邻数量
SCORE_THRESHOLD = 0.4  # 相似度阈值

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

# --------------------- 自定义FAISS封装 ---------------------

class CustomFAISS(FAISS):
    """使用余弦相似度的自定义FAISS封装类"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, any]] = None,
        fetch_k: int = 20,
        **kwargs: any,
    ) -> List[tuple[Document, float]]:
        """使用余弦相似度进行搜索并设置元数据"""
        embedding = self.embedding_function.embed_query(query)
        # 归一化查询向量
        embedding_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_np)
        
        # 直接使用余弦相似度（内积）
        scores, indices = self.index.search(embedding_np, k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                continue
            doc = self.docstore.search(self.index_to_docstore_id[i])
            # 直接使用FAISS返回的内积分数（即余弦相似度）
            similarity = scores[0][j]
            
            # 创建文档副本并设置元数据（关键修复）
            doc_with_metadata = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()  # 复制原始元数据
            )
            # 添加相似度分数到元数据
            doc_with_metadata.metadata["score"] = similarity
            
            # 构建来源信息（使用书名和篇名）
            book_name = doc_with_metadata.metadata.get("书名", "")
            chapter_name = doc_with_metadata.metadata.get("篇名", "")
            if book_name and chapter_name:
                doc_with_metadata.metadata["source"] = f"{book_name}·{chapter_name}"
            elif book_name:
                doc_with_metadata.metadata["source"] = book_name
            elif chapter_name:
                doc_with_metadata.metadata["source"] = chapter_name
            else:
                doc_with_metadata.metadata["source"] = "未知"
            
            docs.append((doc_with_metadata, similarity))
        return docs

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs,
    ) -> FAISS:
        """重写from_documents方法，使用内积索引"""
        # 创建内积索引（IndexFlatIP）
        embeddings = embedding.embed_documents([doc.page_content for doc in documents])
        index = faiss.IndexFlatIP(len(embeddings[0]))
        
        # 归一化向量以便使用内积计算余弦相似度
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # 创建文档存储
        docstore = cls._build_docstore(documents)
        index_to_id = {i: str(i) for i in range(len(documents))}
        
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            normalize_L2=True,  # 确保向量已归一化
            **kwargs,
        )
    
    def add_documents(self, documents: List[Document], **kwargs):
        """添加文档时进行归一化处理"""
        if not documents:
            return
        
        embeddings = self.embedding_function.embed_documents(
            [doc.page_content for doc in documents]
        )
        
        # 归一化新向量
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        
        # 添加到索引
        starting_index = self.index.ntotal
        self.index.add(embeddings_np)
        
        # 更新文档存储
        for i, doc in enumerate(documents):
            idx = str(starting_index + i)
            self.docstore.add({idx: doc})
            self.index_to_docstore_id[starting_index + i] = idx

# --------------------- 数据加载 ---------------------
def load_segmented_documents() -> List[Document]:
    """从JSON缓存文件加载分割后的文档"""
    docs = []
    
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
            metadata = chunk.get('元数据', {})
            # 确保metadata是字典类型
            if not isinstance(metadata, dict):
                metadata = {}
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
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: str, device: str):
        self.model = SentenceTransformer(model_path, device=device)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

def build_vector_store() -> CustomFAISS:
    """构建FAISS向量库（使用余弦相似度）"""
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    
    print(f"正在加载本地嵌入模型: {LOCAL_EMBEDDING_MODEL_PATH}")
    embedding_model = SentenceTransformerEmbeddings(
        model_path=LOCAL_EMBEDDING_MODEL_PATH,
        device=device
    )
    
    faiss_index_path = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.faiss")
    pkl_index_path = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.pkl")
    
    if os.path.exists(faiss_index_path) and os.path.exists(pkl_index_path):
        print(f"加载现有FAISS索引: {faiss_index_path} 和 {pkl_index_path}")
        try:
            # 使用自定义FAISS类加载索引
            vector_store = CustomFAISS.load_local(
                folder_path=FAISS_INDEX_DIR,
                embeddings=embedding_model,
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True
            )
            print("FAISS索引加载成功")
            print(f"索引文档数: {len(vector_store.index_to_docstore_id)}")
            return vector_store
        except Exception as e:
            print(f"加载FAISS索引失败: {str(e)}，尝试重建索引...")
    
    # 创建新索引
    docs = load_segmented_documents()
    if not docs:
        raise ValueError(f"未找到任何文档，请检查 {JSON_CACHE_FILE}")
    
    # 过滤掉过短的文档
    docs = [doc for doc in docs if len(doc.page_content) > 50]
    print(f"创建新FAISS索引，有效文档数量: {len(docs)}")
    
    try:
        # 使用自定义方法创建索引（内积索引）
        vector_store = CustomFAISS.from_documents(
            documents=docs, 
            embedding=embedding_model
        )
        print("FAISS索引创建成功（使用余弦相似度）")
    except Exception as e:
        print(f"标准方法创建索引失败: {str(e)}，尝试分批次方法...")
        # 分批次处理文档
        batch_size = 1000
        vector_store = None
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            
            if vector_store is None:
                # 第一批文档创建索引
                vector_store = CustomFAISS.from_documents(
                    documents=batch_docs, 
                    embedding=embedding_model
                )
                print(f"已创建初始索引，处理文档: {min(i+batch_size, len(docs))}/{len(docs)}")
            else:
                # 后续批次添加文档
                vector_store.add_documents(batch_docs)
                print(f"已添加文档: {min(i+batch_size, len(docs))}/{len(docs)}")
    
    # 保存索引
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
        use_fast=False
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
        max_new_tokens=500,  # 增加生成长度以容纳完整回答
        temperature=0.2,    # 降低随机性
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id  # 避免警告
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. 构建基础检索器
    print("构建基础检索器...")
    retriever = vector_store.as_retriever(
        search_type="similarity",  # 使用相似度搜索
        search_kwargs={
            "k": NEAREST_NEIGHBORS,
            "score_threshold": SCORE_THRESHOLD
        }
    )
    
    # 4. 改进的中医问答专用提示模板
    prompt_template = """
    ### 系统角色：
    你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户问题。
    
    ### 检索到的古籍原文：
    {context}
    
    ### 用户问题：
    {question}
    
    ### 回答要求：
    1. 严格基于提供的古籍原文内容回答
    2. 回答需准确、简洁、专业
    3. 涉及药方需完整列出药材和剂量
    4. 如原文无直接答案，可基于中医理论合理推断
    5. 如无相关古籍原文，请明确告知并给出中医常识解释
    
    ### 回答格式：
    [古籍名称]载：相关古籍原文
    
    ### 回答内容：
    """
    prompt = PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"]
    )
    
    # 5. 创建问答链
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": False  # 关闭详细日志
        },
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
            result = qa_chain.invoke({"query": query})
            
            print("\nAI解答 >>")
            print(result["result"].strip())
            
            source_docs = result.get("source_documents", [])
            if source_docs:
                print("\n参考古籍 >>")
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get("source", "未知")
                    score = doc.metadata.get("score", 0)
                    print(f"{i+1}. 来源：{source} (相似度: {score:.4f})")
                    
                    # 当用户请求查看原文时显示完整内容
                    if query.lower() == "source":
                        print(f"   原文片段：{doc.page_content[:300]}...")
            else:
                print("\n⚠️ 未找到相关古籍原文，回答基于中医常识")
            
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