import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import time

CACHE_DIR = "tcm_split_cache"  # 缓存目录（自动创建）

def parse_metadata_and_content(text):
    """提取元数据和章节内容"""
    metadata = {
        '书名': re.search(r'书名：(.*?)\n', text, re.DOTALL).group(1) if re.search(r'书名：(.*?)\n', text, re.DOTALL) else "未知",
        '作者': re.search(r'作者：(.*?)\n', text, re.DOTALL).group(1) if re.search(r'作者：(.*?)\n', text, re.DOTALL) else "未知",
        '朝代': re.search(r'朝代：(.*?)\n', text, re.DOTALL).group(1) if re.search(r'朝代：(.*?)\n', text, re.DOTALL) else "未知",
        '年份': re.search(r'年份：(.*?)\n', text, re.DOTALL).group(1) if re.search(r'年份：(.*?)\n', text, re.DOTALL) else "未知"
    }
    
    # 分割章节（以<篇名>开头，排除第一个空字符串）
    chapters = re.split(r'<篇名>', text)[1:]
    chapter_list = []
    for chapter in chapters:
        # 提取篇名（第一行）
        title_part, content_part = chapter.split('\n', 1)  # 按第一个换行分割篇名和内容
        chapter_title = title_part.strip()
        
        # 清洗内容：去除<目录>、控制字符（如\x07等）及前后空白
        # 修复点：使用[\x00-\x1F]匹配所有ASCII控制字符（非打印字符）
        cleaned_content = re.sub(r'<目录>|[\x00-\x1F]+', '', content_part)  # 修正后的正则表达式
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()  # 合并连续空白
        
        chapter_list.append({
            '篇名': chapter_title,
            '内容': cleaned_content
        })
    return metadata, chapter_list
    
def split_chapter_into_chunks(chapter_content, chunk_size=1000, chunk_overlap=100):
    """将章节内容分割为小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " "],
        length_function=len
    )
    return text_splitter.split_text(chapter_content)

def process_single_file(file_path):
    """处理单个文本文件，返回分割后的文档块列表"""
    with open(file_path, 'r', encoding='GBK',errors='ignore') as f:
        text = f.read()
    
    metadata, chapter_list = parse_metadata_and_content(text)
    document_chunks = []
    
    for idx, chapter in enumerate(chapter_list, 1):
        chunks = split_chapter_into_chunks(chapter['内容'])
        for chunk_idx, chunk in enumerate(chunks, 1):
            doc = {
                '内容': chunk,
                '元数据': {
                    '文件路径': file_path,
                    '书名': metadata['书名'],
                    '作者': metadata['作者'],
                    '朝代': metadata['朝代'],
                    '年份': metadata['年份'],
                    '篇名': chapter['篇名'],
                    '章节编号': idx,
                    '块编号': chunk_idx,
                    '总块数': len(chunks)
                }
            }
            document_chunks.append(doc)
    return document_chunks

def process_directory(input_dir, output_file="split_results.json"):
    """批量处理目录下所有txt文件"""
    all_chunks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"处理文件：{file_path}")
                chunks = process_single_file(file_path)
                all_chunks.extend(chunks)
    
    # 保存结果为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"分割完成，结果保存至：{output_file}")

def save_to_cache(chunks, cache_file):
    """保存分割结果到缓存"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"缓存已更新：{cache_file}")

def load_from_cache(cache_file):
    """从缓存加载分割结果"""
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cache_file_path(input_file):
    """根据输入文件路径生成缓存文件路径"""
    relative_path = os.path.relpath(input_file, start=INPUT_DIR)
    cache_subdir = os.path.join(CACHE_DIR, os.path.dirname(relative_path))
    os.makedirs(cache_subdir, exist_ok=True)
    return os.path.join(cache_subdir, f"{os.path.basename(input_file)}.cache.json")

def need_reprocess(input_file, cache_file):
    """判断是否需要重新处理（文件更新时间或缓存不存在）"""
    if not os.path.exists(cache_file):
        return True
    input_mtime = os.path.getmtime(input_file)
    cache_mtime = os.path.getmtime(cache_file)
    return input_mtime > cache_mtime  # 输入文件更新时间晚于缓存则重新处理

def process_directory_with_cache(input_dir, output_cache=os.path.join(CACHE_DIR, "all_chunks.cache.json")):
    """带缓存的批量处理（仅处理新增/修改文件）"""
    all_chunks = []
    
    # 首先尝试加载全局缓存
    if os.path.exists(output_cache):
        print(f"检测到全局缓存文件：{output_cache}")
        try:
            with open(output_cache, 'r', encoding='utf-8') as f:
                return json.load(f)  # 直接返回缓存结果（如需强制重新处理可注释此行）
        except Exception as e:
            print(f"加载全局缓存失败，重新处理：{e}")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            cache_file = get_cache_file_path(file_path)
            
            if need_reprocess(file_path, cache_file):
                print(f"处理更新文件：{file_path}")
                chunks = process_single_file(file_path)
                save_to_cache(chunks, cache_file)  # 保存单文件缓存
                all_chunks.extend(chunks)
            else:
                print(f"使用缓存文件：{cache_file}")
                all_chunks.extend(load_from_cache(cache_file))  # 加载单文件缓存
    
    # 保存全局合并缓存（可选，用于快速加载所有数据）
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(output_cache, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"全局缓存已生成：{output_cache}")
    return all_chunks

if __name__ == "__main__":
    INPUT_DIR = "TCM-Ancient-Books-master"  # 输入目录
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # 自动创建缓存目录
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 优先使用缓存，仅新增/修改文件重新处理
    start_time = time.time()
    document_chunks = process_directory_with_cache(INPUT_DIR)
    print(f"处理完成，总耗时：{time.time()-start_time:.2f}秒，总块数：{len(document_chunks)}")
    
    # 示例：加载缓存直接使用（后续运行时可直接调用）
    # with open(os.path.join(CACHE_DIR, "all_chunks.cache.json"), 'r', encoding='utf-8') as f:
    #     document_chunks = json.load(f)