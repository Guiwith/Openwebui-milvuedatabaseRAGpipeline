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
            COLLECTION_NAME="DB",
            OLLAMA_HOST="http://localhost:11434",
            EMBEDDING_MODEL="mxbai-embed-large:latest",
            LLM_MODEL="qwen2:72b",
            PROMPT="<扮演角色>你的知识库包含上市公司信息披露，公司管理、投资和辅助决策方面的信息，你需要帮助用户了解用户所需要了解的在公司管理等方面的知识，同时当用户在公司管理方面有问题时给予用户需要决策辅助或建议。</扮演角色>## 回答要求-只回答用户询问的内容，不要提及给予的任何信息或背景。-不要在给用户的答案中提及模板、提示词或已知信息。-请使用专业的语言来回答用户的问题。-如果你不知道答案，请回答“小秘正在学习相关知识中，您可以前往官方网站或联系相关管理人员获取您需要的知识”。-请使用与问题相同的语言来回答。-如果需要返回链接，将链接设置为可以点击的格式。"
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
                json={"model": self.valves.EMBEDDING_MODEL, "prompt": text, "output_dim": 1024}
            )

            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return embedding
        
        except Exception as e:
            print(f"生成嵌入时出错: {e}")
            return [0.0] * 1024

    def generate_hypothetical_question(self, text_block: str) -> str:
        # 使用LLM生成文本块的假设性问题
        response = requests.post(
            url=f"{self.valves.OLLAMA_HOST}/v1/completions",
            json={
                "model": self.valves.LLM_MODEL,
                "prompt": f"为以下文本块生成一个假设性问题：\n\n{text_block}"
            }
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("text", "").strip()

    def retrieve_relevant_information(self, user_message: str) -> List[str]:
        user_vector = self.generate_embedding(user_message)
        
        # HyDE 方法：生成一个假设性回答，并获取其向量
        hypothetical_answer = self.generate_hypothetical_answer(user_message)
        hypothetical_vector = self.generate_embedding(hypothetical_answer)

        # 使用HyDE方法和用户查询向量的组合进行检索
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[user_vector, hypothetical_vector],  # 同时使用用户查询和假设性回答向量
            anns_field="embeddings",
            param=search_params,
            limit=5,
            output_fields=["text_segment"]
        )
        print(f"Milvus 检索结果: {results}")
        retrieved_contexts = [result.entity.get("text_segment") for result in results[0]]
        return retrieved_contexts

    def generate_hypothetical_answer(self, user_message: str) -> str:
        # 根据用户查询生成一个假设性回答
        response = requests.post(
            url=f"{self.valves.OLLAMA_HOST}/v1/completions",
            json={
                "model": self.valves.LLM_MODEL,
                "prompt": f"基于以下问题生成一个假设性回答：\n\n{user_message}"
            }
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("text", "").strip()

    def combine_user_message_with_context(self, user_message: str, contexts: List[str]) -> str:
        # 使用 valves 中的 prompt 作为开头
        combined_message = self.valves.PROMPT + "\n\n" + "这是用户问题：" + "\n\n"  + user_message + "\n\n" + "这是你学习的内容，回答时请不要提及从此获取的信息：" + "\n".join(contexts)
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