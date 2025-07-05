import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import flwr as fl
import numpy as np
import gc
import random



# Data Transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02),
    #transforms.GaussianBlur(kernel_size=3),
    #transforms.RandomErasing(p=0.3),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


def load_model():
    model = timm.create_model('cait_xxs24_224', pretrained=True, drop_rate = 0.1, attn_drop_rate = 0.1)
    num_classes = 5
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        model.reset_classifier(num_classes)
    return model
'''
def load_model():
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze specific parts (safer approach)
    for name, param in model.named_parameters():
        # Unfreeze the classifier head
        if 'head' in name or 'fc' in name:  # Different DeiT variants use different names
            param.requires_grad = True
            continue

        # Unfreeze last N blocks (safer parsing)
        if 'blocks' in name:
            try:
                # Handle both formats: 'blocks.11.attn.qkv.weight' and 'blocks.11.norm1.weight'
                parts = name.split('.')
                block_idx = int(parts[1])  # Get the block number (e.g., 11 from 'blocks.11...')
                if block_idx >= len(model.blocks) - 2:  # Unfreeze last 2 blocks
                    param.requires_grad = True
            except (IndexError, ValueError):
                # Skip if name doesn't match expected pattern
                continue

    return model
'''

'''
def load_model():
    model = timm.create_model('cait_xxs24_224', pretrained=True, drop_rate=0.1, attn_drop_rate=0.1)
    num_classes = 5

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few blocks and head
    for name, param in model.named_parameters():
        if 'blocks.22' in name or 'blocks.23' in name or 'head' in name:  # Last two blocks
            param.requires_grad = True

    # Modify the head
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        model.reset_classifier(num_classes)

    return model
'''

def train(model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    model.to(device)
    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0


        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update weights AFTER clipping

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss, epoch_acc


def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return loss / len(test_loader.dataset), correct / total


class CaiTClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, val_dataset, client_id, num_clients=4, local_epochs=2):
        self.device = torch.device("cuda:0")


        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.4, device=0)

        self.model = load_model().to(self.device)
        self.client_id = client_id
        self.num_clients = num_clients
        self.local_epochs = local_epochs


        self.train_dataset = train_dataset

        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=1,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=1,
            pin_memory=True if torch.cuda.is_available() else False
        )



    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = train(self.model, self.train_loader, self.local_epochs, self.device)

        # Return metrics along with parameters
        metrics = {
            "train_loss": float(loss),
            "train_accuracy": float(accuracy),
            "client_id": self.client_id
        }

        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def main():
    # Load full datasets
    train_dataset_full = datasets.ImageFolder(
        r"C:\Users\Gaming PC\Documents\2 Datasets _Aptos_Idrid_imbalanced\Client1_Train_Set",
        transform=transform_train)
    val_dataset_full = datasets.ImageFolder(
        r"C:\Users\Gaming PC\Documents\2 Datasets _Aptos_Idrid_imbalanced\Client_1_val",
        transform=transform_val)
    # Client configuration (run with different client_id for each client)
    client_id = 0  # Change this (0-3) for each of your 4 clients
    num_clients = 2

    # Initialize client
    client = CaiTClient(
        train_dataset_full,
        val_dataset_full,
        client_id=client_id,
        num_clients=num_clients,
        local_epochs=3,
    )

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8082", client=client)


if __name__ == "__main__":
    main()