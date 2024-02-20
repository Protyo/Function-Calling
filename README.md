# Getting Started

## Requirements
`Python 3.8+`

## Virtual Environments
It's recommended to create two virtual environments, one for the base library with minimal dependencies and one that contains extra packages necessary to run examples. For our purposes, we run `virtualenv base_env` and `virtualenv example_env`.

## Managing API Keys and Environment Variables
We support integration with external APIs such as OpenAI, Mistral, and Gemini. We recommend managing API keys through environment variables and our example scripts follow this pattern. To set this up, first run `pip install python-dotenv` if not already installed. Then, create a `.env` file and set your environment variables for example scripts there. Below is an example of how to then load an environment variable:

```python
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

import os
api_key = os.environ.get("OPENAI_API_KEY")
```


# Conductor

## Usage

The Conductor class is a lightweight wrapper around LLM backends to primarily log information between `instruct` calls. To instantiate the `Conductor` class, we must choose a backend from `openai`, `mistral`, or `ollama` and choose a model, e.g. `gpt-3.5-turbo`. In this example, we'll show how to use the Conductor class to manage hyperparameter tuning on CIFAR-10.

```
from conductor import Conductor

conductor = Conductor(api_key=api_key, system_prompt=system_prompt, backend="openai", model="gpt-3.5-turbo")
```

As our script is running, we can send messages to store in a history that is packaged with any prompts by calling `log` on the conductor.

```
conductor.log({
    "hyperparameters": hyperparameters,
    "results": {
        "test_accuracy": test_accuracy
    }
})
```

To send a prompt to the LLM backend, call `instruct`:

```
response = conductor.instruct(instruction="Based on the training history, suggest new hyperparameter values. \
                                      Respond in JSON with a key for \"hyperparameters\" that only contains the given \
                                      parameters in the history, mapped to the new values you suggest. Don't repeat \
                                      configurations that have previously been tested. Respond only with JSON.")
hyperparameters = response["hyperparameters"]
```

Reference `example.py` for a complete example using pytorch.

# Function Calling

In `gpt.py`, we define lightweight decorators to make it easy to mark python functions for usage by an LLM. To declare a function as callable, add the `@gpt_callable` decorator to the function.

```
@gpt_callable
def get_weather(latitude: float, longitude: float, date: int, time: float):
    """
    Get's the weather for a given lat, long, on a given date at a given time.
    """
    print("It's 72 degrees.")
```

We expose a function `get_callable_functions()` that collects all functions with `@gpt_callable` and formats them for usage with the OpenAI sdk, so you can simply provide the following:

```
tools = get_callable_functions()
chat_completion = chat_completion_request(messages, tools=tools, model=model)
```

Our functions take care of inspecting and collecting variable names, types, generating descriptions, etc. Reference `example_functions.py` for a complete example that covers hyperparameter tuning.
