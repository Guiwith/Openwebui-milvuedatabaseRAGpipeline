# Openwebui-milvuedatabaseRAGpipeline
 Openwebui-milvuedatabaseRAGpipeline

这是一个openwebui的pipeline。
**Doc2DB**用于监控文件夹文件，并将文件内容切割（默认是2000字符切一段），给LLM进行数据清洗，随后导入进Milvus数据库。
**RAGpipeline**用于管道，详细内容可在openwebui管道页面配置。

## 使用手册：

1. 安装依赖
2. 运行Doc2DB.py(需先进入脚本配置参数)
3. 在openwebui中导入pipeline，并配置参数
4. 开始使用吧！

### 目前存在的问题：

未与网络搜索、文件导入功能连接，直接使用会出现错误。
没有监督机制，回复有几率出现bug。
上下文逻辑有问题，有几率出现答非所问或回答别的聊天的内容。

