import os
import faiss
import json
import torch
import numpy as np
import re
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

# --------------------- 问答系统核心类 ---------------------
class MedicalQAChain:
    def __init__(self, vector_store: CustomFAISS, llm: HuggingFacePipeline):
        self.vector_store = vector_store
        self.llm = llm
        
        # 定义专用提示词模板
        self.templates = {
            "medical_relevance": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要判断用户问题是否与中医医药相关。
            
            ### 相关领域定义：
            1. 疾病：包括疾病的症状、病因、诊断、治疗等
            2. 药物：包括草药、药材、药性、药效等
            3. 药方：包括方剂、处方、针灸、推拿、按摩等治疗方法
            4. 中医理论：包括阴阳五行、脏腑经络、气血津液等
            
            ### 用户问题：
            {query}
            
            ### 判断要求：
            1. 如果问题涉及以上任何领域，回答"是"
            2. 如果问题完全不涉及以上领域，回答"否"
            
            ### 回答格式：
            只需回答一个字："是"或"否"
            """,
            
            "question_classification": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要对用户问题进行分类。
            
            ### 分类选项：
            1. 疾病：问题涉及疾病的症状、病因、诊断、治疗等
            2. 药物：问题涉及草药、药材、药性、药效等
            3. 药方：问题涉及方剂、处方、针灸、推拿、按摩等治疗方法
            4. 多领域：问题同时涉及以上多个领域
            
            ### 用户问题：
            {query}
            
            ### 分类要求：
            1. 根据问题内容选择最合适的分类
            2. 如果问题涉及多个领域，选择"多领域"
            
            ### 回答格式：
            只需回答一个词：["疾病", "药物", "药方", "多领域"]
            """,
            
            "tcm_translation": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要将现代医学术语转换为中医术语。
            
            ### 转换要求：
            1. 使用中医特有的表达方式
            2. 保持原意的准确性
            3. 遵循中医经典表述
            
            ### 示例：
            输入：头痛、发热
            输出：头痛、发热
            
            输入：高血压
            输出：眩晕
            
            输入：糖尿病
            输出：消渴
            
            输入：患者感觉头晕，伴有恶心
            输出：头晕，恶心欲呕
            
            输入：失眠多梦
            输出：不寐多梦
            
            ### 待转换内容：
            {term}
            
            ### 回答格式：
            只需输出转换后的中医术语描述
            """,
            
            "disease_symptom": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户关于疾病症状的问题。
            
            ### 检索到的古籍原文：
            {context}
            
            ### 用户问题：
            患者有以下症状：{question}
            
            ### 回答要求：
            1. 严格基于提供的古籍原文内容回答
            2. 将症状与中医疾病对应，并提供中医术语解释
            3. 涉及相关疾病时需说明典型症状和病理机制
            4. 如原文无直接答案，可基于中医理论合理推断
            
            ### 回答格式：
            [古籍名称]记载：相关古籍原文
            中医诊断：中医术语解释
            """,
            
            "disease_name": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户关于疾病名称的问题。
            
            ### 检索到的古籍原文：
            {context}
            
            ### 用户问题：
            {question}
            
            ### 回答要求：
            1. 严格基于提供的古籍原文内容回答
            2. 解释疾病的病因、病机、典型症状和治疗方法
            3. 如原文无直接答案，可基于中医理论合理推断
            
            ### 回答格式：
            [古籍名称]记载：相关古籍原文
            中医解析：详细解释
            """,
            
            "drug": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户关于药物的问题。
            
            ### 检索到的古籍原文：
            {context}
            
            ### 用户问题：
            {question}
            
            ### 回答要求：
            1. 严格基于提供的古籍原文内容回答
            2. 详细说明药物的性味归经、功效主治、用法用量
            3. 如涉及药物配伍，需完整列出
            
            ### 回答格式：
            [古籍名称]记载：相关古籍原文
            药物详解：详细解释
            """,
            
            "prescription": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户关于药方的问题。
            
            ### 检索到的古籍原文：
            {context}
            
            ### 用户问题：
            {question}
            
            ### 回答要求：
            1. 严格基于提供的古籍原文内容回答
            2. 详细说明药方的组成、功效、主治、用法
            3. 如原文无直接答案，可基于中医理论合理推断
            
            ### 回答格式：
            [古籍名称]记载：相关古籍原文
            方剂解析：详细解释
            """,
            
            "multi_topic": """
            ### 系统角色：
            你是一位专业的中医古籍知识专家，需要基于古籍原文回答用户涉及多个领域的问题。
            
            ### 检索到的古籍原文：
            {context}
            
            ### 用户问题：
            {question}
            
            ### 回答要求：
            1. 严格基于提供的古籍原文内容回答
            2. 分别回答疾病、药物、药方相关部分
            3. 保持回答结构清晰，逻辑严谨
            
            ### 回答格式：
            [疾病部分]
            [古籍名称]记载：相关古籍原文
            解析：...
            
            [药物部分]
            [古籍名称]记载：相关古籍原文
            解析：...
            
            [药方部分]
            [古籍名称]记载：相关古籍原文
            解析：...
            """
        }
    
    def call_llm(self, prompt_template: str, input_vars: dict, max_tokens=100, temperature=0.1) -> str:
        """调用LLM并返回结果"""
        try:
            prompt = PromptTemplate(
                template=prompt_template.strip(),
                input_variables=list(input_vars.keys())
            )
            formatted_prompt = prompt.format(**input_vars)
            result = self.llm(formatted_prompt, max_new_tokens=max_tokens, temperature=temperature)
            return result.strip()
        except Exception as e:
            print(f"LLM调用失败: {str(e)}")
            return ""
    
    def is_medical_question(self, query: str) -> bool:
        """使用LLM判断问题是否与医药相关"""
        result = self.call_llm(
            self.templates["medical_relevance"],
            {"query": query},
            max_tokens=10
        )
        return "是" in result
    
    def classify_question(self, query: str) -> str:
        """使用LLM分类问题类型: disease, drug, prescription, multi"""
        result = self.call_llm(
            self.templates["question_classification"],
            {"query": query},
            max_tokens=10
        )
        
        # 解析LLM返回的分类
        if "疾病" in result:
            return "disease"
        elif "药物" in result:
            return "drug"
        elif "药方" in result:
            return "prescription"
        elif "多领域" in result:
            return "multi"
        else:
            return "unknown"
    
    def translate_to_tcm(self, term: str) -> str:
        """使用LLM将术语翻译为中医术语"""
        return self.call_llm(
            self.templates["tcm_translation"],
            {"term": term},
            max_tokens=100
        )
    
    def is_symptom_query(self, query: str) -> bool:
        """使用LLM判断是否是症状查询"""
        # 简化处理，实际应用中可添加专用LLM判断
        return "症状" in query or "表现" in query or "感觉" in query
    
    def retrieve_documents(self, query: str, k: int = NEAREST_NEIGHBORS) -> List[Tuple[Document, float]]:
        """检索相关文档"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def rerank_documents(self, docs_scores: List[Tuple[Document, float]]) -> List[Document]:
        """重排序文档（按相似度分数）"""
        # 按相似度分数降序排序
        sorted_docs = sorted(docs_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]
    
    def generate_answer(self, context_docs: List[Document], query: str, template_type: str) -> str:
        """使用指定模板生成回答"""
        # 合并文档内容
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 获取对应模板
        template = self.templates.get(template_type, self.templates["disease_name"])
        
        # 生成回答
        return self.call_llm(
            template,
            {"context": context, "question": query},
            max_tokens=500
        )
    
    def process_disease(self, query: str) -> str:
        """处理疾病相关查询"""
        if self.is_symptom_query(query):
            # 流程4: 症状查询
            # 使用LLM翻译症状
            tcm_query = self.translate_to_tcm(query)
            
            # 分别用原查询和翻译后的查询检索
            original_docs = self.retrieve_documents(query)
            translated_docs = self.retrieve_documents(tcm_query)
            
            # 合并并重排序
            combined_docs = original_docs + translated_docs
            reranked_docs = self.rerank_documents(combined_docs)
            
            # 生成回答
            return self.generate_answer(reranked_docs, query, "disease_symptom")
        else:
            # 流程4: 疾病名称查询
            tcm_query = self.translate_to_tcm(query)
            
            # 分别用原查询和翻译后的查询检索
            original_docs = self.retrieve_documents(query)
            translated_docs = self.retrieve_documents(tcm_query)
            
            # 合并并重排序
            combined_docs = original_docs + translated_docs
            reranked_docs = self.rerank_documents(combined_docs)
            
            # 进入流程8
            return self.process_final(reranked_docs, query, "disease_name")
    
    def process_drug(self, query: str) -> str:
        """处理药物相关查询"""
        # 流程5: 药物查询
        tcm_query = self.translate_to_tcm(query)
        
        # 分别用原查询和翻译后的查询检索
        original_docs = self.retrieve_documents(query)
        translated_docs = self.retrieve_documents(tcm_query)
        
        # 合并并重排序
        combined_docs = original_docs + translated_docs
        reranked_docs = self.rerank_documents(combined_docs)
        
        # 进入流程8
        return self.process_final(reranked_docs, query, "drug")
    
    def process_prescription(self, query: str) -> str:
        """处理药方相关查询"""
        # 流程6: 药方查询
        docs = self.retrieve_documents(query)
        reranked_docs = self.rerank_documents(docs)
        
        # 进入流程8
        return self.process_final(reranked_docs, query, "prescription")
    
    def extract_components(self, query: str) -> Dict[str, str]:
        """提取问题中的症状、药物、药方成分"""
        # 简化处理，实际应用中可添加专用LLM提取
        components = {"symptom": "", "drug": "", "prescription": ""}
        
        if "症状" in query:
            components["symptom"] = query
        if "药" in query and not ("药方" in query or "处方" in query):
            components["drug"] = query
        if "方" in query or "针灸" in query or "推拿" in query:
            components["prescription"] = query
        
        return components
    
    def process_multi_topic(self, query: str) -> str:
        """处理多主题查询"""
        # 流程7: 多主题查询
        components = self.extract_components(query)
        
        # 分别处理每个组件
        all_docs = []
        
        if components["symptom"]:
            # 处理症状
            tcm_query = self.translate_to_tcm(components["symptom"])
            symptom_docs = self.retrieve_documents(components["symptom"]) + self.retrieve_documents(tcm_query)
            all_docs.extend(symptom_docs)
        
        if components["drug"]:
            # 处理药物
            tcm_query = self.translate_to_tcm(components["drug"])
            drug_docs = self.retrieve_documents(components["drug"]) + self.retrieve_documents(tcm_query)
            all_docs.extend(drug_docs)
        
        if components["prescription"]:
            # 处理药方
            prescription_docs = self.retrieve_documents(components["prescription"])
            all_docs.extend(prescription_docs)
        
        # 重排序所有文档
        reranked_docs = self.rerank_documents(all_docs)
        
        # 进入流程8
        return self.process_final(reranked_docs, query, "multi_topic")
    
    def process_final(self, docs: List[Document], query: str, template_type: str) -> str:
        """流程8: 最终处理（重排序并生成回答）"""
        # 重排序文档
        reranked_docs = self.rerank_documents([(doc, doc.metadata.get("score", 0)) for doc in docs])
        
        # 生成回答
        return self.generate_answer(reranked_docs, query, template_type)
    
    def answer_question(self, query: str) -> str:
        """主问答流程"""
        # 流程1: 使用LLM判断是否与医药相关
        if not self.is_medical_question(query):
            return "对不起，你的问题无关医药，本系统无法回答。"
        
        # 流程3: 使用LLM分类问题类型
        question_type = self.classify_question(query)
        
        # 根据问题类型路由到不同处理流程
        if question_type == "disease":
            return self.process_disease(query)
        elif question_type == "drug":
            return self.process_drug(query)
        elif question_type == "prescription":
            return self.process_prescription(query)
        elif question_type == "multi":
            return self.process_multi_topic(query)
        else:
            return "无法确定问题类型，请重新表述您的问题。"

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
        max_new_tokens=500,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. 创建问答链
    return MedicalQAChain(vector_store, llm)

# --------------------- 终端交互 ---------------------
def interactive_console(qa_system: MedicalQAChain):
    """命令行交互界面"""
    print("\n===== 中医古籍智能问答系统 =====")
    print("输入格式：直接提问（支持中医病症、方剂、药材等问题）")
    print("输入'quit'退出\n")
    
    while True:
        query = input("用户提问 >> ").strip()
        if query.lower() == "quit":
            break
        
        try:
            print("\n正在处理问题...")
            result = qa_system.answer_question(query)
            
            print("\nAI解答 >>")
            print(result.strip())
            print("\n" + "="*50 + "\n")
        
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")

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
        qa_system = initialize_qa_system()
        print("系统初始化成功！")
    except Exception as e:
        print(f"系统初始化失败：{str(e)}")
        print(f"请检查：")
        print(f"1. 嵌入模型是否在 {LOCAL_EMBEDDING_MODEL_PATH}")
        print(f"2. 大模型是否在 {LOCAL_LLM_MODEL_PATH}")
        print(f"3. 缓存文件是否在 {JSON_CACHE_FILE}")
        exit(1)
    
    # 4. 启动交互界面
    interactive_console(qa_system)