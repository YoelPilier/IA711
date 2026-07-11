# %%
from datasets import load_dataset

dataset = load_dataset("puruchinera/ImageNette")
dataset
# %%


# %%

train_data = dataset["train"]
test_data = dataset["validation"]

# %%
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_data[i]["image"])
    plt.title(f"Label: {train_data[i]['label']}")
    plt.axis("off")
plt.show()
# %%
# Aumentación de datos

# Random Rotation
import random
from PIL import Image


primera_imagen = train_data[0]["image"]


class RandomRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):

        angle = random.randint(-self.angle, self.angle)
        return image.rotate(angle)


RandomRotation(45)(primera_imagen)

# %%

# Random Flip Horizontal


class RandomFlipHorizontal:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


RandomFlipHorizontal(0.5)(primera_imagen)

# %%

# Random Flip Vertical


class RandomFlipVertical:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        return image


RandomFlipVertical(0.5)(primera_imagen)

# %%

# Random Crop


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        right = left + self.size
        bottom = top + self.size
        return image.crop((left, top, right, bottom))


RandomCrop(128)(primera_imagen)


# %%

# Resize


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return image.resize((self.size, self.size))


Resize(64)(primera_imagen)

# %%
# Random Deform
import numpy as np


class RandomDeform:
    def __init__(self):
        pass

    def __call__(self, imge):
        # Abre la imagen
        img = imge.copy()
        modificar = random.randint(0, 1)
        originalsize = img.size
        if modificar == 0:
            return img

        # Genera factores de escala aleatorios entre 0.5 y 1.0
        scale_factor_x = random.uniform(0.2, 1.0)
        scale_factor_y = random.uniform(0.2, 1.0)

        # Calcula las nuevas dimensiones
        new_width = int(img.width * scale_factor_x)
        new_height = int(img.height * scale_factor_y)

        # Redimensiona la imagen
        resized_img = img.resize((new_width, new_height))

        # Genera una imagen de ruido del mismo tamaño que la imagen original
        noise = np.random.normal(0, 1, (originalsize[1], originalsize[0], 3))
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise_img = Image.fromarray((noise * 255).astype(np.uint8))

        # Pega la imagen redimensionada en la imagen de ruido
        noise_img.paste(resized_img, (0, 0))

        return noise_img


RandomDeform()(primera_imagen)
# %%


class RandomErase:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            width, height = image.size
            scale = random.uniform(0.1, 0.4)
            erase_size = int(min(width, height) * scale)
            left = random.randint(0, width - erase_size)
            top = random.randint(0, height - erase_size)
            right = left + erase_size
            bottom = top + erase_size
            image = image.copy()

            # Genera una imagen de ruido del tamaño del área borrada
            noise = np.random.normal(0, 1, (erase_size, erase_size, 3))
            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
            noise_img = Image.fromarray((noise * 255).astype(np.uint8))

            # Pega la imagen de ruido en la imagen original
            image.paste(noise_img, (left, top, right, bottom))

        return image


RandomErase(0.5)(primera_imagen)

# %%

# Random Copy Paste


class RandomCopyPaste:
    def __init__(self, p, size):
        self.p = p
        self.size = size

    def __call__(self, image):
        if random.random() < self.p:
            image = image.copy()
            source_location = (
                random.randint(0, image.width - self.size),
                random.randint(0, image.height - self.size),
            )

            paste_location = (
                random.randint(0, image.width - self.size),
                random.randint(0, image.height - self.size),
            )
            image_patch = image.crop(
                (
                    *source_location,
                    source_location[0] + self.size,
                    source_location[1] + self.size,
                )
            )
            image.paste(image_patch, paste_location)

        return image


RandomCopyPaste(0.5, 100)(primera_imagen)

# %%

# ToTensor


import torch


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, image):
        return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0


ToTensor()(primera_imagen)

# %%


# %%

# Normalize

import torch


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        return (tensor - self.mean[:, None, None]) / self.std[:, None, None]


Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(ToTensor()(primera_imagen))

# %%


class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]

    def __call__(self, tensor):
        result = tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)
        return result


Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(ToTensor()(primera_imagen))
)

# %%


def show_batch(batch, denormalize=None):
    if isinstance(batch, (list, tuple)):
        images, labels = batch[0], batch[1]
    else:
        images, labels = batch, None

    n = min(len(images), 9)
    plt.figure(figsize=(12, 12) if n > 8 else (16, 8))

    for i in range(n):
        plt.subplot(3, 3, i + 1) if n > 8 else plt.subplot(2, 4, i + 1)

        img = denormalize(images[i]) if denormalize else images[i]
        img = img.clamp(0, 1).squeeze().permute(1, 2, 0)

        plt.imshow(img.detach().cpu().numpy())

        if labels is not None:
            plt.title(f"Label: {labels[i]}")

        plt.axis("off")
    plt.show()


# %%
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImagenetteData(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")  # never forget
        label = self.dataset[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
size = (128, 128)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.CenterCrop(128),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
train_dataset = ImagenetteData(train_data, transform=train_transform)
test_dataset = ImagenetteData(test_data, transform=test_transform)


# %%

import os

bs = 64
cpus = os.cpu_count()

train_dataloader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=cpus
)
test_dataloader = DataLoader(
    test_dataset, batch_size=bs, shuffle=False, num_workers=cpus
)

x = next(iter(train_dataloader))
print(x[0].shape, x[1].shape)
# show_batch(x, denormalize=Denormalize(mean, std))
# %%

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# %%
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.flat = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(512, 10)

        # initialize weights
        nn.init.kaiming_normal_(self.c1[0].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c1[4].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c2[0].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c2[4].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c3[0].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c3[4].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c4[0].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.c4[4].weight, nonlinearity="relu", mode="fan_out")
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear", mode="fan_out")
        nn.init.zeros_(self.c1[0].bias)
        nn.init.zeros_(self.c1[4].bias)
        nn.init.zeros_(self.c2[0].bias)
        nn.init.zeros_(self.c2[4].bias)
        nn.init.zeros_(self.c3[0].bias)
        nn.init.zeros_(self.c3[4].bias)
        nn.init.zeros_(self.c4[0].bias)
        nn.init.zeros_(self.c4[4].bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


Model()(torch.randn(2, 3, 28, 28)).shape

# %%
from torch import optim
from torchmetrics.functional.classification import accuracy
from tqdm import tqdm, utils

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-2
wd = 1e-4
epochs = 10
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd
)

# %%

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
# %%
