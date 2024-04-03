import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import USPS
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.tensorboard import SummaryWriter


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load USPS dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = USPS(root="./data", train=True, download=True, transform=transform)
test_dataset = USPS(root="./data", train=False, download=True, transform=transform)


# Define the training function
def train(model, optimizer, criterion, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Define the evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return accuracy


# Experiment configurations
experiments = [
    {"batch_size": 32, "optimizer": "SGD"},
    {"batch_size": 64, "optimizer": "Adam"},
    {"batch_size": 128, "optimizer": "RMSprop"},
]

# Run experiments
for i, config in enumerate(experiments):
    print(
        f"Experiment {i + 1}: Batch Size = {config['batch_size']}, Optimizer = {config['optimizer']}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model = CNN()
    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    for epoch in range(5):
        train(model, optimizer, criterion, train_loader, epoch)

    accuracy = evaluate(model, test_loader)
    print(f"Accuracy: {accuracy:.2f}%\n")
