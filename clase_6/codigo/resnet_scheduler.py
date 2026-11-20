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

import random

import numpy as np
import torch
from PIL import Image


primera_imagen = train_data[0]["image"]


# %%
# Funciones auxiliares


def ensure_rgb(image):
    """
    Garantiza que la entrada sea una imagen PIL con tres canales RGB.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Se esperaba una imagen PIL, pero se recibió {type(image).__name__}"
        )

    return image.convert("RGB")


def parse_size(size):
    """
    Convierte:
        128        -> (128, 128)
        (128, 64)  -> (alto=128, ancho=64)
    """

    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size debe ser mayor que cero")

        return size, size

    if len(size) != 2:
        raise ValueError("size debe ser un entero o una tupla (alto, ancho)")

    height, width = int(size[0]), int(size[1])

    if height <= 0 or width <= 0:
        raise ValueError("Las dimensiones deben ser mayores que cero")

    return height, width


# %%
# Random Rotation


class RandomRotation:
    def __init__(self, angle):
        if isinstance(angle, (int, float)):
            if angle < 0:
                raise ValueError("angle debe ser positivo")

            self.min_angle = -float(angle)
            self.max_angle = float(angle)

        else:
            if len(angle) != 2:
                raise ValueError(
                    "angle debe ser un número o una tupla (mínimo, máximo)"
                )

            self.min_angle = float(angle[0])
            self.max_angle = float(angle[1])

            if self.min_angle > self.max_angle:
                raise ValueError("El ángulo mínimo no puede ser mayor que el máximo")

    def __call__(self, image):
        image = ensure_rgb(image)

        angle = random.uniform(
            self.min_angle,
            self.max_angle,
        )

        return image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0),
        )


RandomRotation(45)(primera_imagen)


# %%
# Random Flip Horizontal


class RandomFlipHorizontal:
    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")

        self.p = p

    def __call__(self, image):
        image = ensure_rgb(image)

        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return image


RandomFlipHorizontal(0.5)(primera_imagen)


# %%
# Random Flip Vertical


class RandomFlipVertical:
    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")

        self.p = p

    def __call__(self, image):
        image = ensure_rgb(image)

        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return image


RandomFlipVertical(0.5)(primera_imagen)


# %%
# Random Crop


class RandomCrop:
    def __init__(self, size):
        self.crop_height, self.crop_width = parse_size(size)

    def __call__(self, image):
        image = ensure_rgb(image)

        width, height = image.size

        # Si la imagen es más pequeña que el recorte,
        # se amplía manteniendo su relación de aspecto.
        if width < self.crop_width or height < self.crop_height:
            scale = max(
                self.crop_width / width,
                self.crop_height / height,
            )

            new_width = max(
                self.crop_width,
                round(width * scale),
            )

            new_height = max(
                self.crop_height,
                round(height * scale),
            )

            image = image.resize(
                (new_width, new_height),
                resample=Image.Resampling.BILINEAR,
            )

            width, height = image.size

        left = random.randint(
            0,
            width - self.crop_width,
        )

        top = random.randint(
            0,
            height - self.crop_height,
        )

        right = left + self.crop_width
        bottom = top + self.crop_height

        return image.crop((left, top, right, bottom))


RandomCrop(128)(primera_imagen)


# %%
# Resize


class Resize:
    def __init__(self, size):
        self.height, self.width = parse_size(size)

    def __call__(self, image):
        image = ensure_rgb(image)

        # PIL recibe el tamaño como (ancho, alto).
        return image.resize(
            (self.width, self.height),
            resample=Image.Resampling.BILINEAR,
        )


Resize(64)(primera_imagen)


# %%
# Random Deform


class RandomDeform:
    def __init__(
        self,
        p=0.5,
        min_scale=0.2,
        max_scale=1.0,
    ):
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")

        if not 0 < min_scale <= max_scale <= 1:
            raise ValueError("Debe cumplirse 0 < min_scale <= max_scale <= 1")

        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image):
        image = ensure_rgb(image)

        if random.random() >= self.p:
            return image.copy()

        width, height = image.size

        scale_factor_x = random.uniform(
            self.min_scale,
            self.max_scale,
        )

        scale_factor_y = random.uniform(
            self.min_scale,
            self.max_scale,
        )

        # max(1, ...) evita crear imágenes con dimensión cero.
        new_width = max(
            1,
            round(width * scale_factor_x),
        )

        new_height = max(
            1,
            round(height * scale_factor_y),
        )

        resized_image = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        # Crea un fondo de ruido con el tamaño original.
        noise = np.random.randint(
            low=0,
            high=256,
            size=(height, width, 3),
            dtype=np.uint8,
        )

        noise_image = Image.fromarray(noise)

        # Posición aleatoria donde se pegará la imagen deformada.
        left = random.randint(
            0,
            width - new_width,
        )

        top = random.randint(
            0,
            height - new_height,
        )

        noise_image.paste(
            resized_image,
            (left, top),
        )

        return noise_image


RandomDeform()(primera_imagen)


# %%
# Random Erase


class RandomErase:
    def __init__(
        self,
        p,
        scale=(0.1, 0.4),
    ):
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")

        if len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
            raise ValueError("scale debe cumplir 0 < mínimo <= máximo <= 1")

        self.p = p
        self.scale = scale

    def __call__(self, image):
        image = ensure_rgb(image)

        if random.random() >= self.p:
            return image

        image = image.copy()

        width, height = image.size

        scale = random.uniform(
            self.scale[0],
            self.scale[1],
        )

        erase_size = round(min(width, height) * scale)

        # Garantiza un tamaño válido.
        erase_size = max(
            1,
            min(erase_size, width, height),
        )

        left = random.randint(
            0,
            width - erase_size,
        )

        top = random.randint(
            0,
            height - erase_size,
        )

        noise = np.random.randint(
            low=0,
            high=256,
            size=(erase_size, erase_size, 3),
            dtype=np.uint8,
        )

        noise_image = Image.fromarray(noise)

        image.paste(
            noise_image,
            (left, top),
        )

        return image


RandomErase(0.5)(primera_imagen)


# %%
# Random Copy Paste


class RandomCopyPaste:
    def __init__(self, p, size):
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")

        self.p = p
        self.patch_height, self.patch_width = parse_size(size)

    def __call__(self, image):
        image = ensure_rgb(image)

        if random.random() >= self.p:
            return image

        image = image.copy()

        width, height = image.size

        # Si el parche solicitado es mayor que la imagen,
        # se limita al tamaño de la imagen.
        patch_width = min(
            self.patch_width,
            width,
        )

        patch_height = min(
            self.patch_height,
            height,
        )

        source_left = random.randint(
            0,
            width - patch_width,
        )

        source_top = random.randint(
            0,
            height - patch_height,
        )

        paste_left = random.randint(
            0,
            width - patch_width,
        )

        paste_top = random.randint(
            0,
            height - patch_height,
        )

        image_patch = image.crop(
            (
                source_left,
                source_top,
                source_left + patch_width,
                source_top + patch_height,
            )
        )

        image.paste(
            image_patch,
            (paste_left, paste_top),
        )

        return image


RandomCopyPaste(0.5, 100)(primera_imagen)


# %%
# ToTensor


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, image):
        image = ensure_rgb(image)

        array = np.array(
            image,
            dtype=np.float32,
            copy=True,
        )

        array = array / 255.0

        tensor = torch.from_numpy(array)

        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        return tensor.contiguous()


ToTensor()(primera_imagen)


# %%
# Normalize


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
        ).view(-1, 1, 1)

        self.std = torch.as_tensor(
            std,
            dtype=torch.float32,
        ).view(-1, 1, 1)

        if self.mean.shape != self.std.shape:
            raise ValueError("mean y std deben tener la misma cantidad de valores")

        if torch.any(self.std <= 0):
            raise ValueError("Los valores de std deben ser mayores que cero")

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Normalize esperaba un tensor de PyTorch")

        if tensor.ndim != 3:
            raise ValueError("El tensor debe tener forma (C, H, W)")

        if tensor.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"El tensor tiene {tensor.shape[0]} canales, "
                f"pero mean y std tienen {self.mean.shape[0]} valores"
            )

        mean = self.mean.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )

        std = self.std.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )

        return (tensor - mean) / std


Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
)(ToTensor()(primera_imagen))


# %%
# Denormalize


class Denormalize:
    def __init__(
        self,
        mean,
        std,
        clamp=False,
    ):
        self.mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
        ).view(-1, 1, 1)

        self.std = torch.as_tensor(
            std,
            dtype=torch.float32,
        ).view(-1, 1, 1)

        if self.mean.shape != self.std.shape:
            raise ValueError("mean y std deben tener la misma cantidad de valores")

        if torch.any(self.std <= 0):
            raise ValueError("Los valores de std deben ser mayores que cero")

        self.clamp = clamp

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Denormalize esperaba un tensor de PyTorch")

        if tensor.ndim != 3:
            raise ValueError("El tensor debe tener forma (C, H, W)")

        if tensor.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"El tensor tiene {tensor.shape[0]} canales, "
                f"pero mean y std tienen {self.mean.shape[0]} valores"
            )

        mean = self.mean.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )

        std = self.std.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )

        result = tensor * std + mean

        if self.clamp:
            result = result.clamp(0.0, 1.0)

        return result


tensor = ToTensor()(primera_imagen)

normalized = Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
)(tensor)

restored = Denormalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    clamp=True,
)(normalized)

restored
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
size = (224, 224)


train_transform = transforms.Compose(
    [
        RandomCrop(size),
        RandomFlipHorizontal(0.4),
        RandomFlipVertical(0.4),
        RandomRotation(20),
        RandomErase(0.2),
        RandomCopyPaste(0.2, 32),
        Resize(size),
        ToTensor(),
        Normalize(mean, std),
    ]
)


test_transform = transforms.Compose(
    [
        Resize(size),
        ToTensor(),
        Normalize(mean, std),
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
show_batch(x, denormalize=Denormalize(mean, std))
# %%

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# %%
import torch.nn as nn
import torch


class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


Stem()(torch.randn(2, 3, 224, 224)).shape


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(
            self.conv1[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.conv2[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if not isinstance(self.shortcut, nn.Identity):
            nn.init.kaiming_normal_(
                self.shortcut[0].weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(identity)
        return self.relu(x)


ResidualBlock(128, 64, 2)(torch.randn(2, 128, 56, 56)).shape


# %%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.l1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.l2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128)
        )
        self.l3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256)
        )
        self.l4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512)
        )
        self.flat = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Linear(512, 10)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


Model()(torch.randn(2, 3, 28, 28)).shape

# %%
from torch import optim
from torchmetrics.functional.classification import accuracy
from tqdm import tqdm, utils

model = Model().to(device)


# %%

loss_fn = nn.CrossEntropyLoss()
lr = 1e-2
epochs = 70
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs
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
        scheduler.step()

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
        f"Test Acc: {test_acc[-1]:.4f}|"
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )
# %%
