import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import flwr as fl
import numpy as np

# Define data transforms
transform_train = transforms.Compose([
   # transforms.RandomResizedCrop(224),
   # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load CaiT model and modify for 5 classes
def load_model():
    model_name = 'cait_s24_224'
    model = timm.create_model(model_name, pretrained=True)

    # Modify for 5 classes
    num_classes = 5
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        in_features = model.get_classifier().in_features
        model.reset_classifier(num_classes)

    return model


# Training function
def train(model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

    return model


# Evaluation function
def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    loss /= len(test_loader.dataset)
    accuracy = correct / total

    return loss, accuracy


# Create a Flower client
class CaiTClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, val_dataset, local_epochs=1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model()
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        # Return model parameters as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy arrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Train the model on local dataset
        self.model = train(self.model, self.train_loader, epochs=self.local_epochs, device=self.device)

        # Return updated model parameters and metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"accuracy": 0.0}

    def evaluate(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Evaluate the model on local validation dataset
        loss, accuracy = test(self.model, self.val_loader, self.device)

        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}


def partition_data(dataset, num_clients, client_id):
    """Partition dataset for federated learning simulation."""
    # Determine the size of each partition
    total_size = len(dataset)
    size_per_client = total_size // num_clients

    # Create partition indices
    indices = list(range(total_size))
    start_idx = client_id * size_per_client
    end_idx = (client_id + 1) * size_per_client if client_id < num_clients - 1 else total_size

    from torch.utils.data import Subset
    return Subset(dataset, indices[start_idx:end_idx])

# Main client execution code
def main():
    # Load full datasets
    train_dataset_full = datasets.ImageFolder(r"/content/drive/MyDrive/ModelTest/train", transform=transform_train)
    val_dataset_full = datasets.ImageFolder(r"/content/drive/MyDrive/ModelTest/val", transform=transform_val)

    # Client ID (change this for each client)
    client_id = 0  # 0 for first client, 1 for second, etc.
    num_clients = 1  # Total number of clients

    # Partition datasets
    train_dataset = partition_data(train_dataset_full, num_clients, client_id)
    val_dataset = partition_data(val_dataset_full, num_clients, client_id)

    # Start Flower client
    client = CaiTClient(train_dataset, val_dataset, local_epochs=2)
    fl.client.start_numpy_client(server_address="4.tcp.ngrok.io:11484", client=client)

if __name__ == "__main__":
    main()
