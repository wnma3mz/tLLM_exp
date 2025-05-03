# coding=utf-8
from typing import List 
import requests

class LLM:
    def __init__(self, name: str, system: str, host_url: str):
        self.name = name
        self.system = system
        self.host_url = host_url
        self.history_messages = []

    def clear(self):
        self.history_messages = []

    def ask(self, user_input: str, url: str) -> str:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user_input}]
        response = requests.post(url, json={"messages": messages, "model": self.name, "stream": False, "max_tokens": 200, "force_prompt": "<think></think>"})
        return response.json()["choices"][0]["message"]["content"].strip()

def main():
    system_prompt = """
你是一个非常聪明的AI聊天机器人，现在你将与其他的AI进行对话，它也和你一样。

你们的目标是：{objective}，你的观点是：{view}

你们的对话将会持续进行，直到你们都确定并同意结束讨论。
""".strip()
    llm1 = LLM("LLM1", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有鸡"), "http://localhost:8022/v1/chat/completions")
    llm2 = LLM("LLM2", system_prompt.format(objective="先有鸡还是先有蛋？", view="先有蛋"), "http://localhost:8022/v1/chat/completions")

    llm1_response = llm1.ask("你好", llm2.host_url)
    print("LLM1: " + llm1_response)
    llm2_response = llm2.ask(llm1_response, llm1.host_url)
    print("LLM2: " + llm2_response)
    llm1_response = llm1.ask(llm2_response, llm2.host_url)
    print("LLM1: " + llm1_response)



if __name__ == "__main__":
    main()