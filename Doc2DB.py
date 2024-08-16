import os
import time
import threading
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import requests
from docx import Document
from PyPDF2 import PdfFileReader

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置Milvus数据库连接
def connect_to_milvus(host='localhost', port='19530', collection_name='VectorRAGDB'):
    try:
        connections.connect(alias="default", host=host, port=port)
        
        # 定义Collection的schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text_segment", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields=fields, description="RAG Database")
        
        if collection_name not in utility.list_collections():
            collection = Collection(name=collection_name, schema=schema)
            logging.info(f"Created new collection: {collection_name}")
        else:
            collection = Collection(name=collection_name)
            logging.info(f"Using existing collection: {collection_name}")
        
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 100},
            "metric_type": "L2"
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        collection.load()
        
        return collection
    except Exception as e:
        logging.error(f"Error connecting to Milvus or creating collection: {e}")
        return None

# LLM总结优化，prompt修改第一行就行，其他的会自动同步
def summarize_text_with_llm(text, llm_model="qwen2:72b", base_url="http://localhost:11434", prompt="以下文段是将要录入RAG系统知识库的文段，请将文段中的所有内容整理一下，整理后的内容需要包含原文段的所有信息，不要出现丢失或者混淆，回答只包含处理过的文段内容，不要返回任何无关的其他文字或语句"):
    try:
        json_input = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "stream": False
        }
        response = requests.post(
            url=f"{base_url}/v1/chat/completions",
            json=json_input
        )
        logging.debug(f"LLM Request Body: {json_input}")
        response.raise_for_status()
        summarized_text = response.json().get("choices", [])[0].get("message", {}).get("content", "")
        return summarized_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error summarizing text with LLM: {e}")
        return text

# 生成嵌入向量
def generate_embedding(text, embedding_model="theepicdev/nomic-embed-text:v1.5-q6_K", base_url="http://localhost:11434"):
    try:
        json_input = {"model": embedding_model, "prompt": text}
        response = requests.post(
            url=f"{base_url}/api/embeddings",
            json=json_input
        )
        logging.debug(f"Embedding Request Body: {json_input}")
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        return embedding
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating embedding: {e}")
        return []

# 文件处理
def process_file(file_path, segment_length=2000, prompt="优化以下文段内容，去除无效信息", collection=None):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".pdf":
            pdf = PdfFileReader(open(file_path, "rb"))
            text = "\n".join([page.extract_text() for page in pdf.pages])
        else:
            logging.warning(f"Unsupported file format: {ext}")
            return

        segments = [text[i:i + segment_length] for i in range(0, len(text), segment_length)]

        for segment in segments:
            if len(segment.strip()) == 0:
                continue

            summarized_segment = summarize_text_with_llm(segment, prompt=prompt)
            embedding = generate_embedding(summarized_segment)

            if embedding and isinstance(embedding, list):
                try:
                    data_to_insert = [[embedding], [summarized_segment]]
                    collection.insert(data_to_insert)
                    logging.info(f"Successfully inserted segment: {summarized_segment[:50]}")
                except Exception as e:
                    logging.error(f"Error inserting data into Milvus: {e}")
            else:
                logging.warning(f"Failed to generate valid embedding for segment: {summarized_segment[:50]}")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

# 文件系统事件处理
class FileHandler(FileSystemEventHandler):
    def __init__(self, collection):
        self.collection = collection
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.txt', '.doc', '.docx', '.pdf')):
            logging.info(f"New file detected: {event.src_path}")
            file_size = 0
            while file_size == 0:
                file_size = os.path.getsize(event.src_path)
                if file_size == 0:
                    logging.info(f"Waiting for file to be written: {event.src_path}")
                    time.sleep(1)
            
            process_file(event.src_path, 2000, "优化以下文段内容，去除无效信息", self.collection)

# 监视目录
def monitor_directory(directory, collection):
    event_handler = FileHandler(collection)
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=False)
    observer.start()
    logging.info(f"Monitoring directory: {directory}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    collection = connect_to_milvus()

    if collection:
        directory_to_watch = "/PATH/TO/YOUR/FILE"#修改为储存txt，pdf,doc文件夹
        monitor_directory(directory_to_watch, collection)
    else:
        logging.error("Failed to connect to Milvus or create the collection.")
