import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# Function to Create Dataloaders
from torch.utils.data import DataLoader
def create_dataloaders(dataset_class,
                       root: str,
                       train_transforms,
                       test_transforms,
                       batch_size: int,
                       num_workers: int,
                       download: bool = True,
                       **kwargs):
    train_data = dataset_class(root=root,
                               train=True,
                               transform=train_transforms,
                               download=download,
                               **kwargs)

    test_data = dataset_class(root=root,
                              train=False,
                              transform=test_transforms,
                              download=download,
                              **kwargs)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# Function to Plot loss curves
def plot_metrics(history, title_suffix=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Epoch {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to Train the model
def train_step(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = running_loss / len(dataloader)
    acc = correct / size
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {100 * acc:.2f}%")
    return avg_loss, acc


# Test Loop function 
def test_loop(dataloader, model, loss_fn,device):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = test_loss / len(dataloader)
    acc = correct / size
    print(f"Val Loss: {avg_loss:.4f}, Accuracy: {100 * acc:.2f}%")
    return avg_loss, acc

# Function to run the train and test def
def train_and_evaluate(epochs, scheduler,
                               train_dataloader, test_dataloader,
                                 model, loss_fn, optimizer, device):
    history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
    }

    best_val_acc = 0.0
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------------")
        train_loss, train_acc = train_step(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_acc = test_loop(test_dataloader, model=model, loss_fn=loss_fn, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model saved! Accuracy: {best_val_acc:.2f}")

        scheduler.step()

    plot_metrics(history, title_suffix=" (Custom CNN on CIFAR-10)")
    return history

def save_model(model, path="final_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"📦 Final model saved to {path}")