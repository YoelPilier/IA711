# %%
from datasets import load_dataset

dataset = load_dataset("ylecun/mnist")
dataset

# %%

train_dataset = dataset["train"]
test_dataset = dataset["test"]


print(train_dataset[0]["label"])
train_dataset[0]["image"]

# %%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        label = self.dataset[idx]["label"]
        return image, label


train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_data = MNISTDataset(train_dataset, transform=train_transform)
test_data = MNISTDataset(test_dataset, transform=test_transform)

train_data[0][0].max(), train_data[0][0].min()

# %%
import os

cpus = os.cpu_count()
bs = 128
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=cpus)

test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=cpus)

batch = next(iter(train_loader))
batch[0].shape, batch[1].shape

# %%
from matplotlib import pyplot as plt

images, labels = batch
images = images[:8]
labels = labels[:8]

# mostrar las imágenes
# cols = 8
# rows = (len(images) + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 4), squeeze=False)

# for i, ax in enumerate(axes.ravel()):
#     ax.axis("off")
#     if i < len(images):
#         image = images[i].detach().cpu().squeeze(0)
#         ax.imshow(image, cmap="gray", vmin=0, vmax=1)
#         ax.set_title(f"Etiqueta: {labels[i].item()}")
#
# fig.tight_layout()
# plt.show()

# %%
import torch
from torch import nn
from torch.nn import init
from torchmetrics.classification import Accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modelo con ReLU
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = nn.Conv2d(1, 64, kernel_size=3, stride=4, padding=1, bias=True)

    def forward(self, x):
        x = self.l1(x)
        return x


MNISTModel()(torch.randn(1, 1, 28, 28)).shape

# %%
# import utils

epochs = 20
lr = 1e-3

model = MNISTModel().to(device)
# model = SigmoidMNISTModel().to(device)
# monitor = utils.ModelMonitor(model)

accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)

Loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)


train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # monitor.step()
        # break
        # break
        train_losses.append(train_loss)
    train_loss /= len(train_loader)
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = Loss(outputs, labels)
            test_loss += loss.item()
            correct += accuracy_metric(outputs, labels.int()).item()
            test_losses.append(test_loss)

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
# %%
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Train and Test Loss")
plt.legend()
plt.show()
# monitor.report()
