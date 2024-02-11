from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json


"""
TODO:
* Support Ollama
"""


def convert_openai(messages):
    return messages

def convert_mistral(messages):
    return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]


backends = {
    "openai": OpenAI,
    "mistral": MistralClient
}


converters = {
    "openai": convert_openai,
    "mistral": convert_mistral
}

chats = {
    "openai": lambda x: x.chat.completions.create,
    "mistral": lambda x: x.chat
}

class Conductor:
    def __init__(self, api_key, system_prompt, model="gpt-3.5-turbo", backend="openai"):
        """

        """
        
        self.client = backends[backend](api_key=api_key)
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.logs = []

    def log(self, status):
        self.logs.append(status)

    def instruct(self, instruction):
        prompt = self.system_prompt
        for log in self.logs:
            prompt += str(log)
        prompt += instruction

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        messages = converters[self.backend](messages=messages)
        response = self._generate(messages)
        dictionary = json.loads(response)
        return dictionary

    def _generate(self, messages):
        chat_response = chats[self.backend](self.client)(
            model=self.model,
            messages=messages,
        )
        content = chat_response.choices[0].message.content
        return content
