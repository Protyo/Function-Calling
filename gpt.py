import functools
import inspect
import ollama
from termcolor import colored
from pprint import pprint

gpt_functions = {}


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


def get_human_readable_typename(type_obj):
    human_readable_types = {
        'int': 'number',
        'str': 'string',
        'float': 'number',
        'bool': 'boolean',
        # Add more types as needed
    }
    return human_readable_types.get(type_obj.__name__, type_obj.__name__)


def gen_param_description(function_name, function_doc, param_name, param_type):
    prompt = "Given a function called {0} with the below docstring:\n{1}\n \
        Reply with a concise description for the parameter named {2}, given its type is {3}.".format(
            function_name,
            function_doc,
            param_name,
            param_type)
    messages = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]
    response = ollama.chat(model='openhermes', messages=messages)
    messages.append({
        "role": "assistant",
        "content": response['message']['content']
    })
    return response['message']['content']


def gpt_callable(func):
    # Add function to the registry
    gpt_functions[func.__name__] = func

    @functools.wraps(func)  # This line is crucial
    def wrapper(*args, **kwargs):
        """Wrapper function's docstring"""
        return func(*args, **kwargs)
    return wrapper


def get_callable_functions():
    funcs = []
    for _, func in gpt_functions.items():
        func_template = {
            "type": "function"
        }
        func_template["function"] = {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        sig = inspect.signature(func)
        params = sig.parameters

        for name, param in params.items():
            param_type = getattr(param.annotation, '__name__', None)
            if param_type is None: continue

            is_required = param.default is param.empty
            if is_required:
                func_template["function"]["parameters"]["required"].append(name)
            
            readable_type = get_human_readable_typename(param.annotation)
            readable_description = gen_param_description(func.__name__, func.__doc__, name, param_type)

            func_template["function"]["parameters"]["properties"][name] = {
                "type": readable_type,
                "description": readable_description
            }
        funcs.append(func_template)
    return funcs


def get_callable_function(name):
    return gpt_functions[name]

# Example usage
if __name__ == '__main__':
    @gpt_callable
    def get_weather(latitude: float, longitude: float, date: int, time: float):
        """
        Get's the weather for a given lat, long, on a given date at a given time.
        """
        print("It's 72 degrees.")

    # Execute all decorated functions
    funcs = get_callable_functions()
    print(funcs)
