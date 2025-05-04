# coding=utf-8
from typing import List 
import requests

class LLM:
    def __init__(self, name: str, system: str, host_url: str, api_key: str):
        self.name = name
        self.system = system
        self.host_url = host_url
        self.history_messages = [{"role": "system", "content": system}]
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def clear_message(self):
        self.history_messages = [{"role": "system", "content": self.system}]

    def add_message(self, role: str, content: str):
        self.history_messages.append({"role": role, "content": content})

    def ask(self, user_input: str, url: str) -> str:
        self.add_message("user", user_input)
        messages = self.history_messages.copy()
        response = requests.post(url, json={"messages": messages, "model": self.name, "stream": False, "max_tokens": 512, "force_prompt": "<think>\n</think>\n\n"}, headers=self.headers)
        return response.json()["choices"][0]["message"]["content"] # .strip()

def main():
    system_prompt = """
你是一个非常聪明的AI聊天机器人，现在你将与其他的AI进行对话，它也和你一样。

你们的目标是：{objective}，你的观点是：{view}

你们的对话将会持续进行，直到你们都确定并同意结束讨论。每次回复只能在 200 token以内。
""".strip()
    # llm1 = LLM("LLM1", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有鸡"), "http://localhost:8022/v1/chat/completions")
    # llm2 = LLM("LLM2", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有蛋"), "http://localhost:8022/v1/chat/completions")
    llm1 = LLM("deepseek-chat", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有鸡"), "https://api.deepseek.com/chat/completions")
    llm2 = LLM("deepseek-chat", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有蛋"), "https://api.deepseek.com/chat/completions")

    # 第一轮分别获取双方首轮 Question
    first_question = "由你开始发表观点"
    llm1_response = llm1.ask(first_question, llm2.host_url)
    print("LLM1 response: " + llm1_response)
    llm2_response = llm2.ask(llm1_response, llm1.host_url)
    print("LLM2 get response " + llm2_response)

    # 第二轮开始后，可以循环处理
    llm1.clear_message()
    llm2.clear_message()
    llm2.add_message("user", llm1_response)
    llm2.add_message("assistant", llm2_response)

    while True:
        llm1_response = llm1.ask(llm2_response, llm2.host_url)
        print("LLM1: " + llm1_response)
        llm1.add_message("assistant", llm1_response)
        llm2_response = llm2.ask(llm1_response, llm1.host_url)
        llm2.add_message("assistant", llm2_response)
        print("LLM2: " + llm2_response)


if __name__ == "__main__":
    main()