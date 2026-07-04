# %%
from datasets import load_dataset

dataset = load_dataset("ylecun/mnist")
dataset
# %%

test_data = dataset["test"]
# %%

train_data = dataset["train"]

# %%
from matplotlib import pyplot as plt


# %%
# %%
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class BinaryMNIST(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


ttransform = transforms.Compose([transforms.ToTensor()])
train_dataset = BinaryMNIST(train_data, transform=ttransform)
test_dataset = BinaryMNIST(test_data, transform=ttransform)

bs = 64
# %%
import os

cpus = os.cpu_count()

train_dataloader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=cpus
)
test_dataloader = DataLoader(
    test_dataset, batch_size=bs, shuffle=False, num_workers=cpus
)

x = next(iter(train_dataloader))
print(x[0].shape, x[1].shape)
# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# %%
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.n1 = nn.BatchNorm2d(64)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.n2 = nn.BatchNorm2d(128)
        self.r2 = nn.ReLU()
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.n3 = nn.BatchNorm2d(256)
        self.r3 = nn.ReLU()
        self.fc = nn.Linear(256, 10)
        # initialize weights
        nn.init.kaiming_normal_(self.c1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.c2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.c3.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.c1.bias)
        nn.init.zeros_(self.c2.bias)
        nn.init.zeros_(self.c3.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.c1(x)
        x = self.n1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.n2(x)
        x = self.r2(x)
        x = self.c3(x)
        x = self.n3(x)
        x = self.r3(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x


Model()(torch.randn(1, 1, 28, 28)).shape
# %%
from torch import optim
from torchmetrics.functional.classification import accuracy
from tqdm import tqdm

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-2
epochs = 10
# model_monitor = utils.ModelMonitor(model)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        x, y = x.to(device), y.to(device).float()

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = loss_fn(out, y.long())
        acc = accuracy(out, y.int(), task="multiclass", num_classes=10)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # model_monitor.step()
    train_loss.append(epoch_loss / len(train_dataloader))
    train_acc.append(epoch_acc / len(train_dataloader))

    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.inference_mode():
        for x, y in tqdm(
            test_dataloader, desc=f"Evaluating Epoch {epoch + 1}/{epochs}"
        ):
            x, y = x.to(device), y.to(device).float()
            out = model(x).squeeze()
            loss = loss_fn(out, y.long())

            acc = accuracy(out, y.int(), task="multiclass", num_classes=10)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    test_loss.append(epoch_loss / len(test_dataloader))
    test_acc.append(epoch_acc / len(test_dataloader))
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss[-1]:.4f} | "
        f"Train Acc: {train_acc[-1]:.4f} | "
        f"Test Loss: {test_loss[-1]:.4f} | "
        f"Test Acc: {test_acc[-1]:.4f}"
    )

# model_monitor.report(include_history=True)
# %%
