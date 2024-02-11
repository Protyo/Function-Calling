from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
import ollama


def identity(x):
    return x

def convert_mistral(messages):
    return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

def parse_ollama(response):
    return response['message']['content']

def parse_openai(response):
    return response.choices[0].message.content

backends = {
    "openai": lambda x: OpenAI(api_key=x),
    "mistral": lambda x: MistralClient(api_key=x),
    "ollama": lambda x: ollama
}

converters = {
    "openai": identity,
    "mistral": convert_mistral,
    "ollama": identity
}

chats = {
    "openai": lambda x: x.chat.completions.create,
    "mistral": lambda x: x.chat,
    "ollama": lambda x: x.chat
}

parsers = {
    "openai": parse_openai,
    "mistral": parse_openai,
    "ollama": parse_ollama
}


class Conductor:
    def __init__(self, api_key, system_prompt, model="gpt-3.5-turbo", backend="openai"):
        """

        """
        
        self.client = backends[backend](api_key)
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
        messages = converters[self.backend](messages)
        response = self._generate(messages)
        dictionary = json.loads(response)
        return dictionary

    def _generate(self, messages):
        chat_response = chats[self.backend](self.client)(
            model=self.model,
            messages=messages,
        )
        content = parsers[self.backend](chat_response)
        return content
