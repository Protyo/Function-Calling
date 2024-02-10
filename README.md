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
