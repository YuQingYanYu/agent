from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 加载.env文件配置
load_dotenv()

# 初始化大模型对象
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0  # 输出更稳定，适合Agent场景
)

if __name__ == "__main__":
    print("=== 金融客服助手已启动 ===")
    print("输入你想问的问题，输入 exit 退出程序\n")

    # 初始化消息列表：只放系统提示词，后续每轮对话都追加进去
    messages = [
        SystemMessage(content="你是一个专业的金融客服助手，回答简洁专业，通俗易懂。")
    ]

    # 开启交互式循环
    while True:
        # 接收用户输入
        user_input = input("你：")

        # 退出判断
        if user_input.strip().lower() in ["exit", "quit", "退出"]:
            print("=== 对话结束 ===")
            break

        # 把用户问题加入对话历史
        messages.append(HumanMessage(content=user_input))

        # 调用大模型
        response = llm.invoke(messages)

        # 把AI回复也加入对话历史（这样下一轮AI能记住之前的内容）
        messages.append(response)

        # 打印回复
        print("AI：", response.content)
        print("-" * 50)
