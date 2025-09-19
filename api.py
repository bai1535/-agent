import os
import faiss
import json
import torch
import numpy as np
import re
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Optional, Dict, Tuple, Union
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
from flask import Flask, request, jsonify
from hashlib import md5

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

# API配置
API_HOST = "0.0.0.0"  # 监听所有接口
API_PORT = 6006       # 服务端口

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

# --------------------- 问答系统核心类 ---------------------
class MedicalQAChain:
    def __init__(self, vector_store: CustomFAISS, llm: HuggingFacePipeline):
        self.vector_store = vector_store
        self.llm = llm
        
        # 加载文档数据构建BM25索引
        print(f"加载文档数据构建BM25索引: {JSON_CACHE_FILE}")
        self.bm25_corpus = []  # 保存所有文档的文本内容
        self.bm25_metadata = []  # 保存所有文档的元数据
        
        try:
            with open(JSON_CACHE_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # 构建BM25语料库
            self.bm25_corpus = [chunk['内容'] for chunk in chunks]
            self.bm25_metadata = [chunk['元数据'] for chunk in chunks]
            
            # 对语料库进行分词（使用jieba）
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_corpus]
            
            # 创建BM25索引
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"BM25索引构建完成，文档数: {len(self.bm25_corpus)}")
        except Exception as e:
            print(f"构建BM25索引失败: {str(e)}")
            self.bm25 = None
        
        # 定义专用提示词模板 - 优化提示工程
        self.templates = {
            "medical_relevance": """
            ### 任务：
            判断用户问题是否与中医医药相关。
            
            ### 相关领域：
            1. 疾病：症状、病因、诊断、治疗
            2. 药物：草药、药材、药性、药效
            3. 药方：方剂、处方、针灸、推拿
            4. 中医理论：阴阳五行、脏腑经络、气血津液
            
            ### 用户问题：
            {query}
            
            ### 判断要求：
            如果问题涉及以上任何领域，回答"是"，否则回答"否"
            
            ### 回答格式：
            只需回答一个字："是"或"否"
            """,
            
            "tcm_translation": """
            ### 任务：
            将现代医学术语转换为中医术语。
            
            ### 转换要求：
            1. 使用中医经典表述
            2. 保持原意的准确性
            3. 只输出转换后的中医术语，不要有任何其他文字
            4. 不要添加任何解释或说明
            5. 如果无法转换，直接返回原始术语
            
            ### 示例：
            输入：高血压
            输出：眩晕
            输入：糖尿病
            输出：消渴
            输入：失眠多梦
            输出：不寐多梦
            
            ### 待转换内容：
            {term}
            
            ### 重要提示：
            只输出转换后的中医术语，不要有任何其他文字！
            """,
            
            "answer_generation": """
            ### 角色定位：中医古籍文献解读者（仅限已检索原文范围内解析）
            ### 应用背景：中医药学术研究辅助/古籍临床应用指导
            ### 任务说明：基于指定古籍原文进行专业解读，禁止添加原文外知识
            
            ### 古籍原文依据：
            {context} （注：该内容为唯一知识来源，不得调用外部知识库）
            
            ### 用户问题：
            {question}
            
            ### 解析规范：
            1. 原文引用需完整标注篇目及段落（如《伤寒论·辨太阳病脉证并治上》第3条）
            2. 专业解释需结合中医基础理论（阴阳、六经、卫气营血等体系）
            3. 结构必须包含【古籍来源】【理论解析】【临床启示】三部分
            
            ### 示例样本：
            当原文为"太阳病，头痛发热，汗出恶风者，桂枝汤主之"
            问题为"桂枝汤适用何种证候"时：
            【古籍来源】《伤寒论》第12条：太阳病，头痛发热，汗出恶风者，桂枝汤主之
            【理论解析】太阳中风证，属营卫不和，卫强营弱，治宜解肌发表、调和营卫
            【临床启示】适用于外感风寒表虚证，症见自汗出、恶风、脉浮缓者
            
            ### 使用约束：
            1. 原文未涉及内容必须明确说明"原文无相关记载"
            2. 禁止进行现代医学解读，仅限中医理论体系阐释
            3. 术语使用需与所引古籍保持时代一致性
            
            ### 回答格式：
            【古籍来源】（请从context中提取古籍名称和篇目）
            【原文重现】（请从context中提取原文内容）
            【中医解析】结合理论体系的专业解读（分点论述时使用中医术语）
            """
        }
    
    def call_llm(self, prompt_template: str, input_vars: dict, max_tokens=1024, temperature=0.1, stop_sequences=None) -> str:
        """调用LLM并返回结果"""
        try:
            prompt = PromptTemplate(
                template=prompt_template.strip(),
                input_variables=list(input_vars.keys()))
            formatted_prompt = prompt.format(**input_vars)
            
            # 添加详细日志
            print(f"\n{'='*50}\nLLM输入提示 (长度: {len(formatted_prompt)}):")
            print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
            
            # 设置生成参数
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,  # 温度大于0时使用采样
                "top_p": 0.9 if temperature > 0 else 1.0
            }
            
            # 添加停止序列
            if stop_sequences:
                generate_kwargs["stop_sequences"] = stop_sequences
            
            result = self.llm(
                formatted_prompt,
                **generate_kwargs
            )
            
            # 清理结果
            cleaned_result = result.strip()
            print(f"\nLLM输出结果 (长度: {len(cleaned_result)}):")
            print(cleaned_result[:500] + "..." if len(cleaned_result) > 500 else cleaned_result)
            print("="*50)
            
            return cleaned_result
        except Exception as e:
            print(f"LLM调用失败: {str(e)}")
            return ""
    
    def is_medical_question(self, query: str) -> bool:
        """使用LLM判断问题是否与医药相关"""
        result = self.call_llm(
            self.templates["medical_relevance"],
            {"query": query},
            max_tokens=10,
            temperature=0  # 使用零温度确保确定输出
        )
        return "是" in result
    
    def translate_to_tcm(self, term: str) -> str:
        """使用LLM将术语翻译为中医术语"""
        # 使用零温度和停止序列确保只输出术语
        result = self.call_llm(
            self.templates["tcm_translation"],
            {"term": term},
            max_tokens=20,
            temperature=0,  # 使用零温度确保确定输出
            stop_sequences=["\n", "输入：", "输出："]  # 添加停止序列防止额外生成
        )
        
        # 清理输出：只保留第一行（如果有换行）
        return result.split("\n")[0].strip()
    
    def retrieve_documents(self, query: str, k: int = NEAREST_NEIGHBORS) -> List[Tuple[Document, float]]:
        """检索相关文档（向量检索）"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def bm25_retrieve(self, query: str, k: int = NEAREST_NEIGHBORS) -> List[Tuple[Document, float]]:
        """使用BM25检索相关文档"""
        if self.bm25 is None:
            print("BM25索引未初始化，无法检索")
            return []
        
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取文档分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top k文档
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            # 创建Document对象
            doc = Document(
                page_content=self.bm25_corpus[idx],
                metadata=self.bm25_metadata[idx].copy()  # 复制元数据
            )
            
            # 设置来源（同向量检索）
            book_name = doc.metadata.get("书名", "")
            chapter_name = doc.metadata.get("篇名", "")
            if book_name and chapter_name:
                doc.metadata["source"] = f"{book_name}·{chapter_name}"
            elif book_name:
                doc.metadata["source"] = book_name
            elif chapter_name:
                doc.metadata["source"] = chapter_name
            else:
                doc.metadata["source"] = "未知"
                
            # 添加BM25分数
            doc.metadata["bm25_score"] = scores[idx]
            
            results.append((doc, scores[idx]))
            
        return results
    
    def rerank_documents(self, docs_scores: List[Tuple[Document, float]]) -> List[Document]:
        """重排序文档（按相似度分数）"""
        # 按相似度分数降序排序
        sorted_docs = sorted(docs_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]
    
    def hybrid_rerank(self, 
                     vec_results: List[Tuple[Document, float]], 
                     bm25_results: List[Tuple[Document, float]], 
                     top_k: int = 2) -> List[Document]:
        """
        混合重排序文档
        使用RRF（Reciprocal Rank Fusion）算法融合向量检索和BM25检索结果
        """
        # 用于文档去重的函数
        def get_doc_id(doc: Document) -> str:
            """生成文档唯一ID（基于内容）"""
            return md5(doc.page_content.encode('utf-8')).hexdigest()
        
        # 构建文档到排名的映射
        doc_ranks = {}
        
        # 处理向量检索结果
        for rank, (doc, score) in enumerate(vec_results, start=1):
            doc_id = get_doc_id(doc)
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {
                    "doc": doc,
                    "vec_rank": rank,
                    "bm25_rank": None
                }
            else:
                doc_ranks[doc_id]["vec_rank"] = rank
        
        # 处理BM25检索结果
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            doc_id = get_doc_id(doc)
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {
                    "doc": doc,
                    "vec_rank": None,
                    "bm25_rank": rank
                }
            else:
                doc_ranks[doc_id]["bm25_rank"] = rank
        
        # 计算RRF分数
        k = 60  # RRF常数，用于平滑排名
        rrf_scores = []
        for doc_id, info in doc_ranks.items():
            score = 0.0
            if info["vec_rank"] is not None:
                score += 1.0 / (k + info["vec_rank"])
            if info["bm25_rank"] is not None:
                score += 1.0 / (k + info["bm25_rank"])
            rrf_scores.append((info["doc"], score))
        
        # 按RRF分数降序排序
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k文档
        return [doc for doc, score in rrf_scores[:top_k]]
    
    def generate_answer(self, context_docs: List[Document], query: str) -> str:
        """生成回答"""
        # 不再过滤文档，直接使用传入的文档
        filtered_docs = context_docs
        
        # 更好的上下文管理
        context = ""
        char_count = 0
        max_context_length = 2500  # 减少上下文长度
        
        for i, doc in enumerate(filtered_docs):
            # 只取每个文档的关键部分
            content = doc.page_content
            if len(content) > 400:
                content = content[:200] + "..." + content[-200:]
            
            source = doc.metadata.get("source", "未知来源")
            entry = f"【{source}】{content}\n\n"
            
            if char_count + len(entry) > max_context_length:
                break
                
            context += entry
            char_count += len(entry)
        
        # 添加上下文摘要
        context += f"\n以上是来自{len(filtered_docs)}个古籍来源的相关内容摘要。"
        
        # 生成回答
        return self.call_llm(
            self.templates["answer_generation"],
            {"context": context, "question": query},
            max_tokens=1024,
            temperature=0.3
        )
    
    def answer_question(self, query: str) -> str:
        """主问答流程"""
        print(f"\n{'#'*50}\n处理问题: {query}\n{'#'*50}")
        
        # 1. 判断是否医疗问题
        print("判断问题是否与医药相关...")
        if not self.is_medical_question(query):
            return "对不起，你的问题无关医药，本系统无法回答。"
        
        # 2. 翻译为中医术语
        print("翻译为中医术语...")
        tcm_query = self.translate_to_tcm(query)
        print(f"原始问题: '{query}' → 中医术语: '{tcm_query}'")
        
        # 3. 混合检索
        print("进行混合检索（向量+BM25）...")
        # 向量检索（使用原始query）
        vec_results = self.retrieve_documents(query, k=NEAREST_NEIGHBORS)
        print(f"向量检索结果: {len(vec_results)}条")
        
        # BM25检索（使用翻译后的tcm_query）
        bm25_results = self.bm25_retrieve(tcm_query, k=NEAREST_NEIGHBORS)
        print(f"BM25检索结果: {len(bm25_results)}条")
        
        # 混合重排序（返回最相关的2个文档）
        reranked_docs = self.hybrid_rerank(vec_results, bm25_results, top_k=2)
        print(f"混合重排序后选择{len(reranked_docs)}个最相关文档")
        
        # 5. 生成最终答案
        print("生成最终答案...")
        return self.generate_answer(reranked_docs, query)

# --------------------- 系统初始化 ---------------------
def initialize_qa_system():
    """初始化问答系统"""
    # 1. 加载向量数据库
    vector_store = build_vector_store()
    
    # 2. 加载大模型 - 使用本地模型
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    print(f"正在加载本地大模型: {LOCAL_LLM_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_LLM_MODEL_PATH, 
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    # 创建HuggingFace管道 - 优化参数
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,  # 增加token限制
        temperature=0.3,      # 稍微提高创造性
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 测试模型是否能生成内容
    print("测试模型生成能力...")
    test_prompt = "中医认为感冒的主要症状是什么？"
    test_output = llm(test_prompt, max_new_tokens=100)
    print(f"模型测试输出: {test_output}")
    
    # 3. 创建问答链
    return MedicalQAChain(vector_store, llm)

# --------------------- API服务 ---------------------
from datetime import datetime  # 新增导入

app = Flask(__name__)
qa_system = None

# 首页路由
# 首页路由 - 修改为交互式页面
@app.route('/', methods=['GET'])
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>中医古籍问答系统</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 40px auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                overflow: hidden;
            }
            header {
                background: #2c3e50;
                color: white;
                padding: 25px 30px;
                text-align: center;
            }
            h1 {
                font-size: 2.2rem;
                margin-bottom: 8px;
                letter-spacing: 0.5px;
            }
            .subtitle {
                font-size: 1.1rem;
                opacity: 0.85;
                font-weight: 300;
            }
            .status {
                display: inline-block;
                background: #27ae60;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.9rem;
                margin-top: 10px;
            }
            .card {
                padding: 30px;
            }
            .input-area {
                margin-bottom: 30px;
            }
            .input-title {
                font-size: 1.3rem;
                margin-bottom: 15px;
                color: #2c3e50;
                display: flex;
                align-items: center;
            }
            .input-title svg {
                margin-right: 10px;
                width: 24px;
                height: 24px;
                fill: #3498db;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                min-height: 120px;
                font-size: 1.1rem;
                resize: vertical;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
            }
            .button-container {
                display: flex;
                justify-content: flex-end;
                margin-top: 15px;
            }
            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 28px;
                font-size: 1.1rem;
                border-radius: 8px;
                cursor: pointer;
                transition: background 0.3s, transform 0.1s;
                display: flex;
                align-items: center;
            }
            button:hover {
                background: #2980b9;
            }
            button:active {
                transform: scale(0.98);
            }
            button svg {
                margin-right: 8px;
            }
            .output-area {
                margin-top: 30px;
                border-top: 1px solid #eee;
                padding-top: 30px;
            }
            .response {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                min-height: 150px;
                border-left: 4px solid #3498db;
                white-space: pre-wrap; /* 保留换行符 */
            }
            .loader {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .loader-dots {
                display: inline-block;
                position: relative;
                width: 80px;
                height: 20px;
            }
            .loader-dots div {
                position: absolute;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #3498db;
                animation-timing-function: cubic-bezier(0, 1, 1, 0);
            }
            .loader-dots div:nth-child(1) {
                left: 8px;
                animation: loader-dots1 0.6s infinite;
            }
            .loader-dots div:nth-child(2) {
                left: 8px;
                animation: loader-dots2 0.6s infinite;
            }
            .loader-dots div:nth-child(3) {
                left: 32px;
                animation: loader-dots2 0.6s infinite;
            }
            .loader-dots div:nth-child(4) {
                left: 56px;
                animation: loader-dots3 0.6s infinite;
            }
            @keyframes loader-dots1 {
                0% { transform: scale(0); }
                100% { transform: scale(1); }
            }
            @keyframes loader-dots3 {
                0% { transform: scale(1); }
                100% { transform: scale(0); }
            }
            @keyframes loader-dots2 {
                0% { transform: translate(0, 0); }
                100% { transform: translate(24px, 0); }
            }
            .error {
                color: #e74c3c;
                background: #fdeded;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                display: none;
            }
            .api-info {
                margin-top: 40px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .api-info h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.5rem;
            }
            .endpoint {
                margin: 15px 0;
                padding: 15px;
                background: #e8f4f8;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }
            .endpoint h3 {
                margin-bottom: 8px;
            }
            code {
                background: #eef2f7;
                padding: 3px 6px;
                border-radius: 4px;
                font-family: monospace;
            }
            pre {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 10px 0;
                font-size: 0.95rem;
            }
            footer {
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 0.9rem;
                border-top: 1px solid #eee;
            }
            @media (max-width: 600px) {
                .container {
                    margin: 20px auto;
                }
                header {
                    padding: 20px 15px;
                }
                h1 {
                    font-size: 1.8rem;
                }
                .card {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>中医古籍问答系统</h1>
                <p class="subtitle">基于传统中医古籍文献的专业知识问答</p>
                <div class="status">系统运行中</div>
            </header>
            
            <div class="card">
                <div class="input-area">
                    <div class="input-title">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                        </svg>
                        <span>请输入您的中医相关问题</span>
                    </div>
                    <textarea id="question" placeholder="例如：感冒的症状有哪些？如何治疗高血压？什么是四君子汤？..."></textarea>
                    <div class="button-container">
                        <button id="ask-btn" type="button">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                                <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
                            </svg>
                            提问
                        </button>
                    </div>
                </div>
                
                <div class="output-area">
                    <div class="input-title">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 9h-2V5h2v6zm0 4h-2v-2h2v2z"/>
                        </svg>
                        <span>系统回答</span>
                    </div>
                    <div class="loader" id="loader">
                        <div class="loader-dots">
                            <div></div>
                            <div></div>
                            <div></div>
                            <div></div>
                        </div>
                        <p>正在检索古籍文献并生成专业回答...</p>
                    </div>
                    <div class="error" id="error"></div>
                    <div class="response" id="response">
                        <p>问题回答将显示在这里。请在上方输入您的中医相关问题并点击"提问"按钮。</p>
                    </div>
                </div>
                
                <div class="api-info">
                    <h2>API 端点</h2>
                    
                    <div class="endpoint">
                        <h3>问答接口</h3>
                        <p><code>POST /ask</code></p>
                        <p>请求示例:</p>
                        <pre>{
    "question": "什么是感冒？"
}</pre>
                    </div>
                    
                    <div class="endpoint">
                        <h3>健康检查</h3>
                        <p><code>GET /health</code></p>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>中医古籍问答系统 v1.0 &copy; 2025</p>
            </footer>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const askBtn = document.getElementById('ask-btn');
                const questionInput = document.getElementById('question');
                const responseArea = document.getElementById('response');
                const loader = document.getElementById('loader');
                const errorArea = document.getElementById('error');
                
                // 添加点击事件监听器
                askBtn.addEventListener('click', function() {
                    submitQuestion();
                });
                
                // 支持按Enter提交（Shift+Enter换行）
                questionInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        submitQuestion();
                    }
                });
                
                function submitQuestion() {
                    const question = questionInput.value.trim();
                    
                    if (!question) {
                        showError('请输入您的问题');
                        return;
                    }
                    
                    // 清除之前的错误和响应
                    errorArea.style.display = 'none';
                    errorArea.textContent = '';
                    responseArea.innerHTML = '<p>正在处理您的问题...</p>';
                    loader.style.display = 'block';
                    
                    // 发送请求到API
                    fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('网络响应异常');
                        }
                        return response.json();
                    })
                    .then(data => {
                        loader.style.display = 'none';
                        if (data.error) {
                            showError(data.error);
                        } else {
                            // 直接显示回答内容
                            responseArea.textContent = data.answer;
                        }
                    })
                    .catch(error => {
                        loader.style.display = 'none';
                        showError('处理问题时出错: ' + error.message);
                    });
                }
                
                function showError(message) {
                    errorArea.textContent = message;
                    errorArea.style.display = 'block';
                    responseArea.innerHTML = '<p>问题回答将显示在这里。请在上方输入您的中医相关问题并点击"提问"按钮。</p>';
                }
            });
        </script>
    </body>
    </html>
    """

# 健康检查
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "vector_db": "active",
            "llm": "active" if qa_system else "inactive",
            "bm25": "active" if qa_system and hasattr(qa_system, 'bm25') and qa_system.bm25 else "inactive"
        }
    })

# favicon 处理
@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回空响应

# 问答接口（保持不变）
@app.route('/ask', methods=['POST'])
def ask_question():
    """API问答接口"""
    data = request.get_json()
    query = data.get('question', '').strip()
    
    if not query:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        print(f"处理问题: {query}")
        result = qa_system.answer_question(query)
        return jsonify({"question": query, "answer": result.strip()})
    except Exception as e:
        return jsonify({"error": f"处理问题时出错: {str(e)}"}), 500

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
        qa_system = initialize_qa_system()
        print("系统初始化成功！")
        
        # 4. 启动API服务
        print(f"启动API服务，访问地址: http://{API_HOST}:{API_PORT}/ask")
        app.run(host=API_HOST, port=API_PORT)
    except Exception as e:
        print(f"系统初始化失败：{str(e)}")
        print(f"请检查：")
        print(f"1. 嵌入模型是否在 {LOCAL_EMBEDDING_MODEL_PATH}")
        print(f"2. 大模型是否在 {LOCAL_LLM_MODEL_PATH}")
        print(f"3. 缓存文件是否在 {JSON_CACHE_FILE}")
        exit(1)