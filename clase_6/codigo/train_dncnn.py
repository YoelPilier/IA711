# %%
from pathlib import Path
import os
import random

import torch
import torch.nn as nn
from accelerate import Accelerator
import datasets
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchvision import transforms
from tqdm import tqdm

# %%
directorio_script = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
directorio_datos = directorio_script / "data"
directorio_datos.mkdir(parents=True, exist_ok=True)

output_dir = directorio_script / "resultados" / "dncnn"
output_dir.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
else:
    mixed_precision = "no"

accelerator = Accelerator(mixed_precision=mixed_precision)
torch.backends.cudnn.benchmark = torch.cuda.is_available()
accelerator.print(f"Dispositivo: {accelerator.device} | Precisión: {mixed_precision}")


# %%
def cargar_dataset_local_o_hub(
    nombre_dataset,
    directorio_base,
    particion="train",
):
    assert directorio_base is not None, "Debe proporcionar el directorio base"

    nombre_carpeta = nombre_dataset.split("/")[1]

    try:
        dataset = datasets.load_from_disk(directorio_base / nombre_carpeta)
        print(f"Dataset cargado desde el directorio local: {nombre_carpeta}")
    except Exception as error:
        print(f"No se pudo cargar el dataset desde el directorio local: {error}")
        print(f"Cargando dataset desde Hugging Face Hub: {nombre_dataset}")
        dataset = datasets.load_dataset(nombre_dataset, split=particion)
        print(f"Guardando dataset en el directorio local: {nombre_carpeta}")
        dataset.save_to_disk(directorio_base / nombre_carpeta)

    return dataset


dataset = cargar_dataset_local_o_hub(
    "puruchinera/ad1k-tagged",
    directorio_datos,
)

dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_data = dataset["train"]
test_data = dataset["test"]


# %%
def ensure_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Se esperaba una imagen PIL, pero se recibió {type(image).__name__}"
        )
    return image.convert("RGB")


class RandomCopyPaste:
    def __init__(self, p, size):
        self.p = p
        self.size = size

    def __call__(self, image):
        image = ensure_rgb(image)
        if random.random() >= self.p:
            return image

        image = image.copy()
        patch_size = min(self.size, image.width, image.height)
        source_left = random.randint(0, image.width - patch_size)
        source_top = random.randint(0, image.height - patch_size)
        paste_left = random.randint(0, image.width - patch_size)
        paste_top = random.randint(0, image.height - patch_size)
        patch = image.crop(
            (
                source_left,
                source_top,
                source_left + patch_size,
                source_top + patch_size,
            )
        )
        image.paste(patch, (paste_left, paste_top))
        return image


# %%
def add_noise(image, noise_factor=None):
    if noise_factor is None:
        noise_factor = 0.1 + torch.rand(1).item()

    noise = noise_factor * torch.normal(
        mean=0.0,
        std=1.0,
        size=image.size(),
    )
    noisy_image = image + noise
    return noisy_image, noise


class NoiseDataset(Dataset):
    def __init__(self, dataset, transform=None, train=True):
        self.dataset = dataset
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.train:
            noisy_image, noise = add_noise(image)
        else:
            noisy_image, noise = add_noise(image, noise_factor=0.2)

        return noisy_image, image, noise


size = 128
mean = [0.67267742, 0.57877398, 0.55262165]
std = [0.30392432, 0.29570012, 0.28292973]


class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


denormalize = Denormalize(mean, std)

train_transform = transforms.Compose(
    [
        RandomCopyPaste(0.5, size // 8),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop(size, pad_if_needed=True),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.0),
            value="random",
        ),
        transforms.Normalize(mean, std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_dataset = NoiseDataset(
    train_data,
    transform=train_transform,
    train=True,
)
test_dataset = NoiseDataset(
    test_data,
    transform=test_transform,
    train=False,
)

# %%
bs = 64
cpus = os.cpu_count()

train_loader = DataLoader(
    train_dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=cpus,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    test_dataset,
    batch_size=bs,
    shuffle=False,
    num_workers=cpus,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# %%
def save_input_batch(batch, filename, n=4):
    noisy, original, noise = batch
    n = min(n, len(noisy))
    plt.figure(figsize=(15, 3 * n))

    for i in range(n):
        noisy_image = denormalize(noisy[i]).clamp(0, 1)
        original_image = denormalize(original[i]).clamp(0, 1)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(noisy_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Imagen con ruido")

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(original_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Original")

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(noise[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Ruido")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


batch = next(iter(train_loader))
save_input_batch(batch, "datos.png")


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels // 4,
                out_channels // 4,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.relu(x + identity)


class DnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.mid_layer = nn.Sequential(*[ResidualBlock(128, 128) for _ in range(13)])
        self.conv3 = nn.Conv2d(
            128,
            3,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mid_layer(x)
        return self.conv3(x)


DnCNN()(torch.randn(1, 3, size, size)).shape


# %%
def initialize(parameter):
    if isinstance(parameter, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(parameter.weight, mode="fan_out", nonlinearity="relu")


model = DnCNN()
model.apply(initialize)

lr = 3e-3
epochs = 10
loss_fn = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    val_loader,
)


# %%
def save_denoising(model, data_loader, filename, n=4):
    if not accelerator.is_main_process:
        return

    model.eval()
    noisy, original, noise = next(iter(data_loader))
    noisy = noisy[:n].to(accelerator.device, non_blocking=True)
    original = original[:n].to(accelerator.device, non_blocking=True)
    noise = noise[:n].to(accelerator.device, non_blocking=True)

    with torch.inference_mode(), accelerator.autocast():
        predicted_noise = model(noisy)

    denoised = denormalize((noisy - predicted_noise).float()).clamp(0, 1).cpu()
    noisy_image = denormalize(noisy.float()).clamp(0, 1).cpu()
    original_image = denormalize(original.float()).clamp(0, 1).cpu()
    noise = noise.float().cpu()
    predicted_noise = predicted_noise.float().cpu()

    n = min(n, len(noisy_image))
    plt.figure(figsize=(20, 4 * n))
    for i in range(n):
        input_psnr = psnr(
            noisy_image[i],
            original_image[i],
            data_range=1.0,
        ).item()
        output_psnr = psnr(
            denoised[i],
            original_image[i],
            data_range=1.0,
        ).item()

        plt.subplot(n, 5, i * 5 + 1)
        plt.imshow(noisy_image[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"Con ruido PSNR: {input_psnr:.2f}")

        plt.subplot(n, 5, i * 5 + 2)
        plt.imshow(original_image[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Original")

        plt.subplot(n, 5, i * 5 + 3)
        plt.imshow(noise[i].clamp(0, 1).permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Ruido")

        plt.subplot(n, 5, i * 5 + 4)
        plt.imshow(predicted_noise[i].clamp(0, 1).permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Ruido predicho")

        plt.subplot(n, 5, i * 5 + 5)
        plt.imshow(denoised[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"Resultado PSNR: {output_psnr:.2f}")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


save_denoising(model, val_loader, "antes_de_entrenar.png")

# %%
train_loss = []
train_psnr = []
val_loss = []
val_psnr = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    for noisy, original, noise in tqdm(
        train_loader,
        desc=f"Entrenamiento época {epoch + 1}/{epochs}",
        disable=not accelerator.is_local_main_process,
    ):
        noisy = noisy.to(accelerator.device, non_blocking=True)
        original = original.to(accelerator.device, non_blocking=True)
        noise = noise.to(accelerator.device, non_blocking=True)
        optimizer.zero_grad()

        with accelerator.autocast():
            predicted_noise = model(noisy)
            loss = loss_fn(predicted_noise, noise)

        accelerator.backward(loss)
        optimizer.step()

        denoised = denormalize((noisy - predicted_noise).float()).clamp(0, 1)
        original_image = denormalize(original.float()).clamp(0, 1)
        psnr_value = psnr(denoised, original_image, data_range=1.0)

        epoch_loss += loss.item()
        epoch_psnr += psnr_value.item()

    train_loss.append(epoch_loss / len(train_loader))
    train_psnr.append(epoch_psnr / len(train_loader))

    model.eval()
    epoch_loss = 0
    epoch_psnr = 0

    with torch.inference_mode():
        for noisy, original, noise in tqdm(
            val_loader,
            desc=f"Validación época {epoch + 1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        ):
            noisy = noisy.to(accelerator.device, non_blocking=True)
            original = original.to(accelerator.device, non_blocking=True)
            noise = noise.to(accelerator.device, non_blocking=True)

            with accelerator.autocast():
                predicted_noise = model(noisy)
                loss = loss_fn(predicted_noise, noise)

            denoised = denormalize((noisy - predicted_noise).float()).clamp(0, 1)
            original_image = denormalize(original.float()).clamp(0, 1)
            psnr_value = psnr(denoised, original_image, data_range=1.0)

            epoch_loss += loss.item()
            epoch_psnr += psnr_value.item()

    val_loss.append(epoch_loss / len(val_loader))
    val_psnr.append(epoch_psnr / len(val_loader))

    accelerator.print(
        f"Época {epoch + 1}/{epochs} | "
        f"Pérdida entrenamiento: {train_loss[-1]:.6f} | "
        f"PSNR entrenamiento: {train_psnr[-1]:.4f} | "
        f"Pérdida validación: {val_loss[-1]:.6f} | "
        f"PSNR validación: {val_psnr[-1]:.4f}"
    )

    save_denoising(
        model,
        val_loader,
        f"epoch_{epoch + 1:03d}.png",
    )
