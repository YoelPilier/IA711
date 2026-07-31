# %%
from pathlib import Path
import os

import torch
import torch.nn as nn
from accelerate import Accelerator
import datasets
from matplotlib import pyplot as plt
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

output_dir = directorio_script / "resultados" / "autoencoder"
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


class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


# %%
class AnimeFaceDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


size = 128
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
denormalize = Denormalize(mean, std)

train_transform = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
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

train_dataset = AnimeFaceDataset(train_data, transform=train_transform)
test_dataset = AnimeFaceDataset(test_data, transform=test_transform)

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
def save_input_batch(batch, filename, n=8):
    n = min(n, len(batch))
    plt.figure(figsize=(16, 8))
    for i in range(n):
        image = denormalize(batch[i]).clamp(0, 1)
        plt.subplot(2, 4, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
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
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(8, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += identity
        return self.relu(out)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 128, stride=1),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(128, 512, stride=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)


print(AutoEncoder().encode((torch.randn(2, 3, size, size))).shape)


# %%


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", a=0.1, nonlinearity="leaky_relu"
        )


model = AutoEncoder()
model.apply(initialize_weights)

lr = 2e-3
epochs = 30

loss_fn = nn.MSELoss().to(accelerator.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5
)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)


# %%
def save_reconstructions(model, data_loader, filename, n=8):
    if not accelerator.is_main_process:
        return

    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    images = next(iter(data_loader))[:n].to(accelerator.device, non_blocking=True)

    with torch.inference_mode(), accelerator.autocast():
        latent = raw_model.encode(images)
        reconstructed = raw_model.decode(latent)

    images = denormalize(images.float()).clamp(0, 1).cpu()
    reconstructed = denormalize(reconstructed.float()).clamp(0, 1).cpu()
    latent = latent[:, :3].float().cpu()
    latent_min = latent.amin(dim=(1, 2, 3), keepdim=True)
    latent_max = latent.amax(dim=(1, 2, 3), keepdim=True)
    latent = (latent - latent_min) / (latent_max - latent_min + 1e-8)

    n = min(n, len(images))
    plt.figure(figsize=(20, 8))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        plt.subplot(3, n, i + n + 1)
        plt.imshow(latent[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Latente")

        plt.subplot(3, n, i + 2 * n + 1)
        plt.imshow(reconstructed[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Reconstrucción")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


save_reconstructions(model, val_loader, "antes_de_entrenar.png")

# %%
train_loss = []
train_psnr = []
val_loss = []
val_psnr = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    for data in tqdm(
        train_loader,
        desc=f"Entrenamiento época {epoch + 1}/{epochs}",
        disable=not accelerator.is_local_main_process,
    ):
        data = data.to(accelerator.device, non_blocking=True)
        optimizer.zero_grad()

        with accelerator.autocast():
            result = model(data)
            loss = loss_fn(result, data)

        accelerator.backward(loss)
        optimizer.step()

        result_image = denormalize(result.float()).clamp(0, 1)
        original_image = denormalize(data.float()).clamp(0, 1)
        psnr_value = psnr(result_image, original_image, data_range=1.0)

        epoch_loss += loss.item()
        epoch_psnr += psnr_value.item()

    train_loss.append(epoch_loss / len(train_loader))
    train_psnr.append(epoch_psnr / len(train_loader))

    model.eval()
    epoch_loss = 0
    epoch_psnr = 0

    with torch.inference_mode():
        for data in tqdm(
            val_loader,
            desc=f"Validación época {epoch + 1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        ):
            data = data.to(accelerator.device, non_blocking=True)
            with accelerator.autocast():
                result = model(data)
                loss = loss_fn(result, data)

            result_image = denormalize(result.float()).clamp(0, 1)
            original_image = denormalize(data.float()).clamp(0, 1)
            psnr_value = psnr(result_image, original_image, data_range=1.0)

            epoch_loss += loss.item()
            epoch_psnr += psnr_value.item()

    val_loss.append(epoch_loss / len(val_loader))
    val_psnr.append(epoch_psnr / len(val_loader))
    actual_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    accelerator.print(
        f"Época {epoch + 1}/{epochs} | "
        f"Pérdida entrenamiento: {train_loss[-1]:.6f} | "
        f"PSNR entrenamiento: {train_psnr[-1]:.4f} | "
        f"Pérdida validación: {val_loss[-1]:.6f} | "
        f"PSNR validación: {val_psnr[-1]:.4f} |"
        f"LR: {actual_lr:.6f}"
    )

    save_reconstructions(
        model,
        val_loader,
        f"epoch_{epoch + 1:03d}.png",
    )

# %%
