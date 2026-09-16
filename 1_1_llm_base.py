from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化客户端（DeepSeek完全兼容OpenAI接口格式）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL")
)
MODEL = os.getenv("MODEL_NAME")

def chat_with_llm(system_prompt: str, user_input: str, temperature: float = 0.0, max_tokens: int = 500):
    """原生LLM调用函数，封装核心参数"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# 测试不同参数效果
if __name__ == "__main__":
    system_prompt = "你是一个专业的气象助手，回答简洁准确。"
    
    print("=== 测试1：Temperature=0（稳定模式）===")
    print(chat_with_llm(system_prompt, "用一句话描述北京的秋天", temperature=0.0))
    
    print("\n=== 测试2：Temperature=1.5（创意模式）===")
    print(chat_with_llm(system_prompt, "用一句话描述北京的秋天", temperature=1.5))
    
    print("\n=== 测试3：Max Tokens=50（限制长度）===")
    print(chat_with_llm(system_prompt, "详细介绍北京的气候特点", max_tokens=100))
