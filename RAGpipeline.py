from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection
from schemas import OpenAIChatMessage


class Pipeline:
    class Valves(BaseModel):
        MILVUS_HOST: str
        MILVUS_PORT: str
        COLLECTION_NAME: str
        OLLAMA_HOST: str
        EMBEDDING_MODEL: str
        LLM_MODEL: str
        PROMPT: str  # 新增自定义 prompt 字段

    def __init__(self):
        self.name = "Ollama Vector Database Pipeline"
        
        # 初始化 Valves 参数，包含 prompt
        self.valves = self.Valves(
            MILVUS_HOST="localhost",
            MILVUS_PORT="19530",
            COLLECTION_NAME="RAGDB",
            OLLAMA_HOST="http://localhost:11434",
            EMBEDDING_MODEL="theepicdev/nomic-embed-text:v1.5-q6_K",
            LLM_MODEL="qwen2:72b",
            PROMPT="你是一个知识丰富的助手，能够回答各种问题。"  # 默认的自定义 prompt
        )
        
        self.connect_to_milvus()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def connect_to_milvus(self):
        connections.connect(alias="default", host=self.valves.MILVUS_HOST, port=self.valves.MILVUS_PORT)
        self.collection = Collection(self.valves.COLLECTION_NAME)
    
    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                url=f"{self.valves.OLLAMA_HOST}/api/embeddings",
                json={"model": self.valves.EMBEDDING_MODEL, "prompt": text}
            )

            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return embedding
        
        except Exception as e:
            print(f"生成嵌入时出错: {e}")
            return [0.0] * 512

    def retrieve_relevant_information(self, user_message: str) -> List[str]:
        user_vector = self.generate_embedding(user_message)
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[user_vector],
            anns_field="embeddings",
            param=search_params,
            limit=5,
            output_fields=["text_segment"]
        )
        print(f"Milvus 检索结果: {results}")
        retrieved_contexts = [result.entity.get("text_segment") for result in results[0]]
        return retrieved_contexts

    def combine_user_message_with_context(self, user_message: str, contexts: List[str]) -> str:
        # 使用 valves 中的 prompt 作为开头
        combined_message = self.valves.PROMPT + "\n\n" + user_message + "\n\n" + "\n".join(contexts)
        return combined_message
    
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        
        # 每次新请求时清空 body['messages']，确保不保留旧的对话上下文
        body['messages'] = []  # 清空旧的上下文
        
        retrieved_contexts = self.retrieve_relevant_information(user_message)
        combined_message = self.combine_user_message_with_context(user_message, retrieved_contexts)
        
        # 添加当前用户消息到 body['messages']
        body['messages'].append({"role": "user", "content": combined_message})
        
        if "user" in body:
            print("######################################")
            print(f'# 用户: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# 消息: {user_message}")
            print("######################################")
        
        try:
            r = requests.post(
                url=f"{self.valves.OLLAMA_HOST}/v1/chat/completions",
                json={**body, "model": self.valves.LLM_MODEL},
                stream=True,
            )
            
            r.raise_for_status()
            
            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
