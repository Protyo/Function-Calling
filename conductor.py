from openai import OpenAI
import json

"""
<system prompt>
<config A>
<config A result>

<config B>
<config B result>

<instruction>
"""


"""
You are a 10x engineer, specifically an expert ML engineer. We are optimizing a model on CIFAR10. Think carefully about hyperparameter selection.

Training run 1: {
    config: {
        "GPUs": 10,
        "RAM": 128,
        "Storage": 12,
        "Bandwidth": "AdamW"
    },
    metrics: {
        requests: {
            1000000
        },
        uploads: {
            100
        }
    }
}


"""


"""
TODO:
* Support Mistral medium
* Support Ollama
"""

class Conductor:
    def __init__(self, api_key, system_prompt, model="gpt-3.5-turbo"):
        """

        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.logs = []

    def log(self, status):
        self.logs.append(status)

    def instruct(self, instruction):
        prompt = self.system_prompt
        for log in self.logs:
            prompt += str(log)
        prompt += instruction
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        content = chat_completion.choices[0].message.content
        dictionary = json.loads(content)
        return dictionary
