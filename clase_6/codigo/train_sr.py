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
from torchvision.models import VGG19_Weights, vgg19
from tqdm import tqdm
from torch.nn.functional import interpolate

# %%
directorio_script = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
directorio_datos = directorio_script / "data"
directorio_datos.mkdir(parents=True, exist_ok=True)

output_dir = directorio_script / "resultados" / "superresolucion"
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


def parse_size(size):
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


class RandomCopyPaste:
    def __init__(self, p, size):
        self.p = p
        self.patch_height, self.patch_width = parse_size(size)

    def __call__(self, image):
        image = ensure_rgb(image)
        if random.random() >= self.p:
            return image

        image = image.copy()
        width, height = image.size
        patch_width = min(self.patch_width, width)
        patch_height = min(self.patch_height, height)
        source_left = random.randint(0, width - patch_width)
        source_top = random.randint(0, height - patch_height)
        paste_left = random.randint(0, width - patch_width)
        paste_top = random.randint(0, height - patch_height)
        patch = image.crop(
            (
                source_left,
                source_top,
                source_left + patch_width,
                source_top + patch_height,
            )
        )
        image.paste(patch, (paste_left, paste_top))
        return image


class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


# %%
def add_gaussian_noise(image):
    noise_factor = random.uniform(0, 0.8)
    noise = noise_factor * torch.normal(
        mean=0.0,
        std=1.0,
        size=image.size(),
    )
    return image + noise


class AnimeDataset(Dataset):
    def __init__(self, dataset, transform=None, size=512, train=True):
        self.dataset = dataset
        self.transform = transform
        self.lowres = transforms.Resize(size // 4, antialias=False)
        self.train = train
        self.crop = transforms.RandomCrop(size, pad_if_needed=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        if self.train:
            image = self.crop(image)
        image = self.transform(image)
        lowres = self.lowres(image)
        if self.train:
            lowres = add_gaussian_noise(lowres)
        return lowres, image


size_train = 128
size_test = 512

mean = [0.67267742, 0.57877398, 0.55262165]
std = [0.30392432, 0.29570012, 0.28292973]

denormalize = Denormalize(mean, std)

train_transform = transforms.Compose(
    [
        RandomCopyPaste(0.5, size_train // 8),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
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
        transforms.Resize(size_test),
        transforms.CenterCrop(size_test),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_dataset = AnimeDataset(
    train_data,
    transform=train_transform,
    size=size_train,
    train=True,
)
test_dataset = AnimeDataset(
    test_data,
    transform=test_transform,
    size=size_test,
    train=False,
)

# %%
bs = 64
val_bs = 32
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
    batch_size=val_bs,
    shuffle=False,
    num_workers=cpus,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# %%
def save_input_batch(batch, filename, n=3):
    lowres, original = batch
    n = min(n, len(lowres))
    plt.figure(figsize=(15, 6))
    for i in range(n):
        lowres_image = denormalize(lowres[i]).clamp(0, 1)
        original_image = denormalize(original[i]).clamp(0, 1)

        plt.subplot(2, n, i + 1)
        plt.imshow(lowres_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Baja resolución")

        plt.subplot(2, n, i + n + 1)
        plt.imshow(original_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Original")

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


class SuperResolution(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(3, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128),
            nn.PixelShuffle(upscale_factor),
            ResidualBlock(32, 128),
            nn.PixelShuffle(upscale_factor),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)) + interpolate(
            x, scale_factor=4, mode="bilinear", align_corners=False
        )


SuperResolution()(torch.randn(1, 3, 32, 32)).shape


# %%
class PerceptualMixedLoss(nn.Module):
    def __init__(self, loss_fn, alpha=0.05, beta=1):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:21].eval()
        for parameter in vgg.parameters():
            parameter.requires_grad = False
        self.vgg = vgg
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        x_features = self.vgg(x)
        with torch.no_grad():
            y_features = self.vgg(y)
        return self.alpha * self.loss_fn(
            x_features,
            y_features,
        ) + self.beta * self.loss_fn(x, y)


# %%
def initialize(parameter):
    if isinstance(parameter, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(parameter.weight, mode="fan_out", nonlinearity="relu")
        if parameter.bias is not None:
            nn.init.zeros_(parameter.bias)


model = SuperResolution()
model.apply(initialize)

lr = 3e-3
epochs = 10
loss_fn = PerceptualMixedLoss(nn.MSELoss()).to(accelerator.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    val_loader,
)


# %%
def save_super_resolution(model, data_loader, filename, n=4):
    if not accelerator.is_main_process:
        return

    model.eval()
    lowres, original = next(iter(data_loader))
    lowres = lowres[:n].to(accelerator.device, non_blocking=True)
    original = original[:n].to(accelerator.device, non_blocking=True)

    with torch.inference_mode(), accelerator.autocast():
        result = model(lowres)

    lowres = denormalize(lowres.float()).clamp(0, 1).cpu()
    result = denormalize(result.float()).clamp(0, 1).cpu()
    original = denormalize(original.float()).clamp(0, 1).cpu()

    n = min(n, len(lowres))
    plt.figure(figsize=(15, 5 * n))
    for i in range(n):
        psnr_value = psnr(result[i], original[i], data_range=1.0).item()

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(lowres[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Baja resolución")

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(result[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"Resultado PSNR: {psnr_value:.2f}")

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(original[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Original")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


save_super_resolution(model, val_loader, "antes_de_entrenar.png")

# %%
train_loss = []
train_psnr = []
val_loss = []
val_psnr = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    for lowres, original in tqdm(
        train_loader,
        desc=f"Entrenamiento época {epoch + 1}/{epochs}",
        disable=not accelerator.is_local_main_process,
    ):
        lowres = lowres.to(accelerator.device, non_blocking=True)
        original = original.to(accelerator.device, non_blocking=True)
        optimizer.zero_grad()

        with accelerator.autocast():
            result = model(lowres)
            loss = loss_fn(result, original)

        accelerator.backward(loss)
        optimizer.step()

        result_image = denormalize(result.float()).clamp(0, 1)
        original_image = denormalize(original.float()).clamp(0, 1)
        psnr_value = psnr(result_image, original_image, data_range=1.0)

        epoch_loss += loss.item()
        epoch_psnr += psnr_value.item()

    train_loss.append(epoch_loss / len(train_loader))
    train_psnr.append(epoch_psnr / len(train_loader))

    model.eval()
    epoch_loss = 0
    epoch_psnr = 0

    with torch.inference_mode():
        for lowres, original in tqdm(
            val_loader,
            desc=f"Validación época {epoch + 1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        ):
            lowres = lowres.to(accelerator.device, non_blocking=True)
            original = original.to(accelerator.device, non_blocking=True)

            with accelerator.autocast():
                result = model(lowres)
                loss = loss_fn(result, original)

            result_image = denormalize(result.float()).clamp(0, 1)
            original_image = denormalize(original.float()).clamp(0, 1)
            psnr_value = psnr(result_image, original_image, data_range=1.0)

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

    save_super_resolution(
        model,
        val_loader,
        f"epoch_{epoch + 1:03d}.png",
    )

# %%
