from conductor import Conductor
from pprint import pprint
from gpt import gpt_callable, get_callable_functions, get_callable_function
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt 

import os
api_key = os.environ.get("OPENAI_API_KEY")

system_prompt = "Act as an ML engineer. You are overseeing the training of a model on CIFAR10 and you are trying to find \
            the best performing hyperparameters."
messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
]

from openai import OpenAI

client = OpenAI(api_key=api_key)

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice="auto", model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def gen(prompt, model="gpt-3.5-turbo"):
    messages.append({
        "role": "user",
        "content": prompt
    })
    tools = get_callable_functions()
    chat_completion = chat_completion_request(messages, tools=tools, model=model)
    while chat_completion.choices[0].finish_reason == 'tool_calls':
        tool_calls = []
        tool_results = []
        for tool_call in chat_completion.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool = get_callable_function(tool_name)
            result = tool(**tool_args)
            tool_calls.append({
                        'id': tool_call.id,
                        'type': 'function',
                        'function': tool_call.function
                    })
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': tool_name,
                'content': result
            })
            
        messages.append({
            'role': 'assistant',
            'tool_calls': tool_calls
        })
        for tool_result in tool_results:
            messages.append(tool_result)
        
        chat_completion = chat_completion_request(messages, tools=tools, model=model)

    content = chat_completion.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": content
    })
    return content


# CIFAR-10 training using a vanilla RESNET
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
device = "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=1)


classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
hyperparameters = {
    "learning_rate": 0.001,
    "momentum": 0.9
}


@gpt_callable
def update_hyperparameter(name: str, new_value: float):
    """Update the hyperparameter to the new value."""
    hyperparameters[name] = new_value
    return "{0} updated to {1}.".format(name, new_value)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch.optim as optim


@gpt_callable
def train():
    """Trains a new model with the current config and returns the test accuracy upon completion."""
    criterion = nn.CrossEntropyLoss()
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=hyperparameters["learning_rate"], momentum=hyperparameters["momentum"])
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct // total
    return 'Accuracy of the network on the 10000 test images: {0}\%'.format(int(test_accuracy))

if __name__ == '__main__':
    for _ in range(1):
        prompt = "Suggest new values based on previous choices and performance. Use your available \
                tools to update parameters and train the model. Available hyperparameters and their initial values: {0}.".format(hyperparameters)
        r = gen(prompt)
        print(r)
        
