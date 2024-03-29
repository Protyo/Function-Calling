from conductor import Conductor
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import os
api_key = os.environ.get("OPENAI_API_KEY")

system_prompt = "Act as an expert ML engineer. You are training a RESNET50 model on CIFAR-10."
conductor = Conductor(api_key=api_key, system_prompt=system_prompt, backend="openai", model="gpt-3.5-turbo")


# CIFAR-10 training using a vanilla RESNET
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
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

    # Hyperparameters
    hyperparameters = {
        "learning_rate": 0.001,
        "momentum": 0.9
    }

    for hyperparameter_run in range(10): 
        print("BEGIN TRAINING")
        print("Configuration:")
        pprint(hyperparameters)
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
        print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')
        conductor.log({
            "hyperparameters": hyperparameters,
            "results": {
                "test_accuracy": test_accuracy
            }
        })
        response = conductor.instruct(instruction="Based on the training history, suggest new hyperparameter values. \
                                      Respond in JSON with a key for \"hyperparameters\" that only contains the given \
                                      parameters in the history, mapped to the new values you suggest. Don't repeat \
                                      configurations that have previously been tested. Respond only with JSON.")
        hyperparameters = response["hyperparameters"]
