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

# Load USPS dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


# Perform one-hot encoding for the targets
def one_hot_encoding(target, num_classes=10):
    return torch.eye(num_classes)[target]


class OneHotUSPS(USPS):
    def __getitem__(self, index):
        img, target = super(OneHotUSPS, self).__getitem__(index)
        return img, one_hot_encoding(target)


train_dataset = OneHotUSPS(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = OneHotUSPS(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16 * 16, 128)  # Adjust input size to match USPS image size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 16)  # Adjust view size to match USPS image size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define CNN architecture
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


# Evaluation function
def evaluate(model, test_loader, writer):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            targets.extend(
                target.max(dim=1)[1].tolist()
            )  # Convert one-hot to class indices
            predictions.extend(pred.flatten().tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    precision = precision_score(targets, predictions, average="weighted")
    recall = recall_score(targets, predictions, average="weighted")
    confusion = confusion_matrix(targets, predictions)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}\n")
    writer.add_scalar("Loss/test", test_loss, len(train_loader) * (epoch + 1))
    writer.add_scalar("Accuracy/test", accuracy, len(train_loader) * (epoch + 1))
    writer.add_scalar("Precision/test", precision, len(train_loader) * (epoch + 1))
    writer.add_scalar("Recall/test", recall, len(train_loader) * (epoch + 1))
    writer.flush()
    return confusion


# Training function
def train(model, optimizer, criterion, train_loader, writer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        # Convert one-hot encoded target to class indices
        _, target_indices = target.max(dim=1)
        loss = criterion(output, target_indices)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {batch_idx*len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item()}"
            )
            writer.add_scalar(
                "Loss/train", loss.item(), len(train_loader) * epoch + batch_idx
            )


# Set up TensorBoard
writer = SummaryWriter()

# Initialize models, optimizer, and loss function
mlp_model = MLP()
cnn_model = CNN()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training and evaluation loop
for epoch in range(5):
    train(mlp_model, optimizer_mlp, criterion, train_loader, writer, epoch)
    train(cnn_model, optimizer_cnn, criterion, train_loader, writer, epoch)

    print("\nMLP Model Evaluation:")
    mlp_confusion = evaluate(mlp_model, test_loader, writer)
    print("\nCNN Model Evaluation:")
    cnn_confusion = evaluate(cnn_model, test_loader, writer)
