from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("BASE_URL"))
MODEL = os.getenv("MODEL_NAME")

# ----------------------
# 1. 定义工具函数
# ----------------------
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    # 模拟天气数据，学习阶段无需接入真实API
    weather_data = {
        "北京": "晴，气温22℃，风力3级，空气质量优",
        "上海": "多云转小雨，气温25℃，风力2级，空气质量良",
        "深圳": "雷阵雨，气温28℃，风力4级，空气质量良"
    }
    return weather_data.get(city, f"暂无{city}的天气数据")

# ----------------------
# 2. 定义工具描述（告诉模型有什么工具可用）
# ----------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询的城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ----------------------
# 3. 完整函数调用流程
# ----------------------
def run_weather_agent(user_query: str):
    # 第一轮对话：让模型判断是否需要调用工具
    messages = [
        {"role": "system", "content": "你是天气助手，需要查询天气时请调用工具，绝对不要编造数据。"},
        {"role": "user", "content": user_query}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 让模型自动决定是否调用工具
    )
    
    response_message = response.choices[0].message
    
    # 判断模型是否返回了工具调用指令
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"[系统日志] 模型决定调用工具：{function_name}")
        print(f"[系统日志] 传入参数：{function_args}")
        
        # 执行工具函数
        if function_name == "get_weather":
            result = get_weather(**function_args)
        
        print(f"[系统日志] 工具返回结果：{result}")
        
        # 把工具结果加入对话历史，第二轮调用模型生成最终回答
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return final_response.choices[0].message.content
    else:
        # 不需要调用工具，直接返回回答
        return response_message.content

# 测试运行
if __name__ == "__main__":
    print("=== 测试1：需要调用工具 ===")
    print(run_weather_agent("北京今天天气怎么样？"))
    
    print("\n=== 测试2：不需要调用工具 ===")
    print(run_weather_agent("什么是雷阵雨？"))
