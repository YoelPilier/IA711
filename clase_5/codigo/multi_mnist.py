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
import utils

# utils.show_batch(x[0][:16], x[1][:16])
# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# %%
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 512)
        self.norm1 = nn.LayerNorm(512)
        self.act1 = nn.Sigmoid()
        self.l2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.act2 = nn.Sigmoid()
        self.l3 = nn.Linear(256, 10)

        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.l3.bias)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.l3(x)
        return x


# Model()(torch.randn(1, 1, 28, 28)).shape
# %%
from torch import optim
from torchmetrics.functional.classification import accuracy
from tqdm import tqdm

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-2
epochs = 10
# model_monitor = utils.ModelMonitor(model)
optimizer = optim.SGD(model.parameters(), lr=lr)

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
