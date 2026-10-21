from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SmallCNN

BASE = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
train_data = datasets.CIFAR10(root=str(BASE / "data"), train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root=str(BASE / "data"), train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
model = SmallCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f} | test_accuracy={correct/total:.4f}")
torch.save({"state_dict": model.cpu().state_dict(), "classes": train_data.classes}, BASE / "cifar10_cnn.pth")
print(f"Saved {BASE / 'cifar10_cnn.pth'}")
