from dotenv import load_dotenv
from conductor import Conductor

load_dotenv()  # take environment variables from .env.

import os
api_key = os.environ.get("OPENAI_API_KEY")

system_prompt = "Act as an expert ML engineer. You are training a RESNET50 model on CIFAR-100. Below is the history of hyperparameters and their resulting loss at 10 epochs. Suggest the next hyperparameter configuration in JSON, but first explain your reasoning step-by-step."
conductor = Conductor(api_key=api_key, system_prompt=system_prompt)


conductor.register("learning_rate", 0.0001)
conductor.register("batch_size", 256)
conductor.register("depth", 12)


for _ in range(100):
    for _ in range(10):
        stats = model.train()
        conductor.log(stats)
    updates = conductor.update()
    model.update(updates)
