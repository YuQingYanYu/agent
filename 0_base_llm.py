from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 加载.env文件里的配置
load_dotenv()

# 初始化大模型对象
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0  # 设为0，输出更稳定，适合Agent场景
)

# 测试运行
if __name__ == "__main__":
    # 构造消息：系统提示词 + 用户问题
    messages = [
        SystemMessage(content="你是一个专业的金融客服助手，回答简洁专业。"),
        HumanMessage(content="你好，简单介绍一下什么是货币基金")
    ]
    
    # 调用大模型
    response = llm.invoke(messages)
    
    # 打印结果
    print("AI回复：", response.content)
