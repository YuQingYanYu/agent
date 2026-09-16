from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("BASE_URL"))
MODEL = os.getenv("MODEL_NAME")

# ----------------------
# 工具集合
# ----------------------
def get_weather(city: str) -> str:
    """查询指定城市的完整天气情况"""
    weather_data = {
        "北京": "晴，气温22℃，风力3级",
        "上海": "多云转小雨，气温25℃，风力2级",
        "广州": "晴，气温30℃，风力2级"
    }
    return weather_data.get(city, f"暂无{city}天气数据")

def get_temperature(city: str) -> str:
    """只查询指定城市的气温数值，用于温度对比"""
    temp_data = {"北京": "22℃", "上海": "25℃", "广州": "30℃"}
    return f"{city}当前气温：{temp_data.get(city, '未知')}"

# 工具名到函数的映射
tool_map = {
    "get_weather": get_weather,
    "get_temperature": get_temperature
}

# ----------------------
# ReAct系统提示词：强制模型按固定格式输出
# ----------------------
REACT_SYSTEM_PROMPT = """
你是一个会逐步思考的智能助手，解决问题时严格按照以下格式输出：

Thought: 在这里写你的思考，分析当前问题，决定下一步做什么
Action: 工具名(参数)
Observation: 工具返回的结果
...（以上循环可以重复多次）

当你认为已经得到足够信息，可以回答用户问题时，输出：
Thought: 我已经有足够信息，可以回答用户问题了
Answer: 你的最终答案

注意：
1. 每次只执行一个Action，不要同时调用多个工具
2. 必须严格遵守输出格式，Thought、Action、Answer各占一行
3. 不要编造Observation，Observation由工具执行后提供
"""

def run_react_agent(user_query: str, max_steps: int = 5):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    for step in range(max_steps):
        print(f"\n----- 第 {step+1} 步 -----")
        
        # 调用模型生成思考与行动
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
        response_text = response.choices[0].message.content.strip()
        print("模型输出：\n" + response_text)
        messages.append({"role": "assistant", "content": response_text})
        
        # 判断是否输出了最终答案
        if "Answer:" in response_text:
            answer = re.search(r"Answer:\s*(.*)", response_text, re.DOTALL).group(1).strip()
            return answer
        
        # 解析Action，提取工具名和参数
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response_text)
        if not action_match:
            return "错误：模型输出格式不正确，无法解析Action"
        
        tool_name = action_match.group(1)
        tool_arg = action_match.group(2).strip().strip('"').strip("'")
        
        # 执行工具，得到观察结果
        if tool_name not in tool_map:
            observation = f"错误：不存在工具{tool_name}"
        else:
            observation = tool_map[tool_name](tool_arg)
        
        print(f"Observation: {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return "已达到最大步数，未能解决问题"

# 测试：需要两步调用才能解决的对比问题
if __name__ == "__main__":
    print("="*50)
    print("测试问题：北京和上海哪个城市气温更高？")
    print("="*50)
    result = run_react_agent("北京和上海哪个城市气温更高？")
    print("\n" + "="*50)
    print("最终答案：", result)
