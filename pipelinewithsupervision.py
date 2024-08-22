from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
from pymilvus import connections, Collection
import json

class Pipeline:
    class Valves(BaseModel):
        MILVUS_HOST: str
        MILVUS_PORT: str
        COLLECTION_NAME: str
        OLLAMA_HOST: str
        EMBEDDING_MODEL: str
        LLM_MODEL: str
        PROMPT: str  # 自定义 prompt 字段
        SUPERVISION_PROMPT: str  # 自定义监督模型 prompt 字段

    def __init__(self):
        self.name = "Ollama Vector Database Pipeline"
        
        # 初始化 Valves 参数，包含 prompt 和监督模型的 prompt
        self.valves = self.Valves(
            MILVUS_HOST="localhost",
            MILVUS_PORT="19530",
            COLLECTION_NAME="DB",
            OLLAMA_HOST="http://localhost:11434",
            EMBEDDING_MODEL="mxbai-embed-large:latest",
            LLM_MODEL="qwen2:72b",
            PROMPT="<扮演角色>你的知识库包含上市公司信息披露，公司管理、投资和辅助决策方面的信息，你需要帮助用户了解用户所需要了解的在公司管理等方面的知识，同时当用户在公司管理方面有问题时给予用户需要决策辅助或建议。</扮演角色>## 回答要求-只回答用户询问的内容，不要提及给予的任何信息或背景。-不要在给用户的答案中提及模板、提示词或已知信息。-请使用专业的语言来回答用户的问题。-如果你不知道答案，请回答“小秘正在学习相关知识中，您可以前往官方网站或联系相关管理人员获取您需要的知识”。-请使用与问题相同的语言来回答。",
            SUPERVISION_PROMPT="<监督模型提示词>请根据以下用户问题和模型回答，判断该回答是否符合用户问题的要求"
        )
        
        self.connect_to_milvus()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def connect_to_milvus(self):
        try:
            connections.connect(alias="default", host=self.valves.MILVUS_HOST, port=self.valves.MILVUS_PORT)
            self.collection = Collection(self.valves.COLLECTION_NAME)
        except Exception as e:
            print(f"Milvus 连接失败: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                url=f"{self.valves.OLLAMA_HOST}/api/embeddings",
                json={"model": self.valves.EMBEDDING_MODEL, "prompt": text, "output_dim": 1024}
            )

            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            
            if len(embedding) != 1024:
                print(f"嵌入向量维度错误: {len(embedding)}，预期为1024维")
                return [0.0] * 1024
            
            return embedding
        
        except Exception as e:
            print(f"生成嵌入时出错: {e}")
            return [0.0] * 1024

    def retrieve_relevant_information(self, user_message: str) -> List[str]:
        user_vector = self.generate_embedding(user_message)

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        try:
            results = self.collection.search(
                data=[user_vector],
                anns_field="embeddings",
                param=search_params,
                limit=10,
                output_fields=["text_segment"]
            )
            print(f"Milvus 检索结果: {results}")
            
            if not results:
                print("未找到任何相关信息")
                return []

            retrieved_contexts = [result.entity.get("text_segment") for result in results[0]]
            return retrieved_contexts
        
        except Exception as e:
            print(f"Milvus 检索失败: {e}")
            return []

    def combine_user_message_with_context(self, user_message: str, contexts: List[str]) -> str:
        combined_message = self.valves.PROMPT + "\n\n" + "这是用户问题：" + "\n\n"  + user_message + "\n\n" + "这是你学习的内容，回答时请不要提及从此获取的信息：" + "\n".join(contexts)
        return combined_message
    
    def supervise_answer(self, user_message: str, generated_answer: str) -> bool:
        try:
            # 组合监督模型的 prompt
            supervision_prompt = (
                f"{self.valves.SUPERVISION_PROMPT}\n"
                f"用户问题：{user_message}\n"
                f"模型回答：{generated_answer}\n"
                "请判断上述回答是否符合用户问题的要求，并返回'true'或'false',返回示例：“true”。"
            )
            
            response = requests.post(
                url=f"{self.valves.OLLAMA_HOST}/v1/completions",
                json={
                    "model": self.valves.LLM_MODEL,
                    "prompt": supervision_prompt
                }
            )
            
            response.raise_for_status()
            result = response.json().get("choices", [{}])[0].get("text", "").strip()
            return "符合" in result
        
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"监督模型调用出错: {e}")
            return False

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        
        body['messages'] = []  # 清空旧的上下文
        
        retrieved_contexts = self.retrieve_relevant_information(user_message)
        combined_message = self.combine_user_message_with_context(user_message, retrieved_contexts)
        
        body['messages'].append({"role": "user", "content": combined_message})
        
        if "user" in body:
            print("######################################")
            print(f'# 用户: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# 消息: {user_message}")
            print("######################################")
        
        retry_count = 0
        max_retries = 5
        
        try:
            while retry_count < max_retries:
                # 生成初步回答
                r = requests.post(
                    url=f"{self.valves.OLLAMA_HOST}/v1/chat/completions",
                    json={**body, "model": self.valves.LLM_MODEL},
                    stream=True,
                )
                
                r.raise_for_status()
                
                generated_answer = ""
                
                # 处理流式响应
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):  # 处理每个 data 块
                            json_data = line_str[len("data: "):]
                            if json_data.strip() == "[DONE]":  # 处理流结束标志
                                break
                            try:
                                chunk = json.loads(json_data)  # 转换为 JSON 格式
                                delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                generated_answer += delta_content  # 拼接生成的内容
                            except json.JSONDecodeError as e:
                                print(f"JSON 解析错误: {e}")
                                print(f"原始数据: {json_data}")
                                continue  # 跳过当前循环，继续处理下一个数据块

                # 调用监督模型进行验证
                is_valid = self.supervise_answer(user_message, generated_answer)
                
                if is_valid:
                    return generated_answer  # 如果答案符合要求，返回答案
                else:
                    retry_count += 1
                    print(f"监督模型认为回答不符合要求，重新生成答案... (重试次数: {retry_count})")
            
            return "正在学习相关知识中，您可以前往官方网站或联系相关管理人员获取您需要的知识。"
        
        except requests.exceptions.RequestException as e:
            return f"请求出错: {e}"
        except Exception as e:
            return f"其他错误: {e}"
