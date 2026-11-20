# %%
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
import datasets
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# %%
directorio_script = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
directorio_datos = directorio_script / "data"
directorio_datos.mkdir(parents=True, exist_ok=True)

output_dir = directorio_script / "resultados" / "vae"
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
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
size = 64


class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


denormalize = Denormalize(mean, std)

train_transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


# %%
class AD1KDataset(Dataset):
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


train_dataset = AD1KDataset(train_data, transform=train_transform)
test_dataset = AD1KDataset(test_data, transform=test_transform)

# %%
bs = 128
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
def save_input_batch(batch, filename, n=32):
    batch = denormalize(batch[:n]).clamp(0, 1).cpu()
    grid = make_grid(batch, nrow=8)
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
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
        self.relu = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
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

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + identity
        return self.relu(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=128, img_size=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_hw = img_size // 8
        embedding_dim = 128 * self.latent_hw * self.latent_hw

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 128),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    128,
                    self.latent_hw,
                    self.latent_hw,
                ),
            ),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            nn.Conv2d(
                64,
                3,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(
            logvar,
            min=-10,
            max=10,
        )
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        z = self.reparameterize(
            mu,
            logvar,
        )

        decoded = self.decoder(self.fc_decode(z))

        return decoded, mu, logvar


print(
    VAE(
        latent_dim=128,
        img_size=size,
    )(
        torch.randn(
            2,
            3,
            size,
            size,
        )
    )[0].shape
)


# %%
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            a=0.1,
            nonlinearity="leaky_relu",
        )

    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def kl_divergence(mean, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl = -0.5 * torch.sum(
        1 + logvar - mean.pow(2) - logvar.exp(),
        dim=1,
    )
    return torch.mean(kl)


class VaeLoss(nn.Module):
    def __init__(self, loss_fn=nn.MSELoss(), alpha=1, beta=1e-3):
        super().__init__()
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, mean, logvar):
        kl = kl_divergence(mean, logvar)
        loss = self.alpha * self.loss_fn(x, y) + self.beta * kl
        return loss, kl


# %%
model = VAE(latent_dim=128, img_size=size)
model.apply(initialize_weights)

lr = 3e-3
epochs = 30
loss_fn = VaeLoss(loss_fn=nn.MSELoss(), alpha=1, beta=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    val_loader,
)


# %%
def save_vae_images(model, data_loader, filename, n=8):
    if not accelerator.is_main_process:
        return

    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    images = next(iter(data_loader))[:n].to(accelerator.device, non_blocking=True)

    with torch.inference_mode(), accelerator.autocast():
        reconstructed, _, _ = raw_model(images)
        z = torch.randn(n, raw_model.latent_dim, device=accelerator.device)
        generated = raw_model.decoder(raw_model.fc_decode(z))

    images = denormalize(images.float()).clamp(0, 1).cpu()
    reconstructed = denormalize(reconstructed.float()).clamp(0, 1).cpu()
    generated = denormalize(generated.float()).clamp(0, 1).cpu()

    n = min(n, len(images))
    plt.figure(figsize=(20, 8))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        plt.subplot(3, n, i + n + 1)
        plt.imshow(reconstructed[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Reconstrucción")

        plt.subplot(3, n, i + 2 * n + 1)
        plt.imshow(generated[i].permute(1, 2, 0).numpy())
        plt.axis("off")
        if i == 0:
            plt.title("Muestra")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


save_vae_images(model, val_loader, "antes_de_entrenar.png")

# %%
train_loss = []
train_psnr = []
train_kl = []
train_latent_mean = []
train_latent_std = []
val_loss = []
val_psnr = []
val_kl = []
val_latent_mean = []
val_latent_std = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_kl = 0
    epoch_latent_mean = 0
    epoch_latent_std = 0

    for data in tqdm(
        train_loader,
        desc=f"Entrenamiento época {epoch + 1}/{epochs}",
        disable=not accelerator.is_local_main_process,
    ):
        data = data.to(accelerator.device, non_blocking=True)
        optimizer.zero_grad()

        with accelerator.autocast():
            reconstructed, mu, logvar = model(data)
            loss, kl = loss_fn(reconstructed, data, mu, logvar)

        accelerator.backward(loss)
        optimizer.step()

        reconstructed_image = denormalize(reconstructed.float()).clamp(0, 1)
        original_image = denormalize(data.float()).clamp(0, 1)
        psnr_value = psnr(reconstructed_image, original_image, data_range=1.0)
        z = accelerator.unwrap_model(model).reparameterize(mu, logvar)

        epoch_loss += loss.item()
        epoch_psnr += psnr_value.item()
        epoch_kl += kl.item()
        epoch_latent_mean += z.mean().item()
        epoch_latent_std += z.std(unbiased=False).item()

    train_loss.append(epoch_loss / len(train_loader))
    train_psnr.append(epoch_psnr / len(train_loader))
    train_kl.append(epoch_kl / len(train_loader))
    train_latent_mean.append(epoch_latent_mean / len(train_loader))
    train_latent_std.append(epoch_latent_std / len(train_loader))

    model.eval()
    epoch_loss = 0
    epoch_psnr = 0
    epoch_kl = 0
    epoch_latent_mean = 0
    epoch_latent_std = 0

    with torch.inference_mode():
        for data in tqdm(
            val_loader,
            desc=f"Validación época {epoch + 1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        ):
            data = data.to(accelerator.device, non_blocking=True)
            with accelerator.autocast():
                reconstructed, mu, logvar = model(data)
                loss, kl = loss_fn(reconstructed, data, mu, logvar)

            reconstructed_image = denormalize(reconstructed.float()).clamp(0, 1)
            original_image = denormalize(data.float()).clamp(0, 1)
            psnr_value = psnr(reconstructed_image, original_image, data_range=1.0)
            z = accelerator.unwrap_model(model).reparameterize(mu, logvar)

            epoch_loss += loss.item()
            epoch_psnr += psnr_value.item()
            epoch_kl += kl.item()
            epoch_latent_mean += z.mean().item()
            epoch_latent_std += z.std(unbiased=False).item()

    val_loss.append(epoch_loss / len(val_loader))
    val_psnr.append(epoch_psnr / len(val_loader))
    val_kl.append(epoch_kl / len(val_loader))
    val_latent_mean.append(epoch_latent_mean / len(val_loader))
    val_latent_std.append(epoch_latent_std / len(val_loader))

    accelerator.print(
        f"Época {epoch + 1}/{epochs} | "
        f"Pérdida entrenamiento: {train_loss[-1]:.6f} | "
        f"PSNR entrenamiento: {train_psnr[-1]:.4f} | "
        f"KL entrenamiento: {train_kl[-1]:.4f} | "
        f"Pérdida validación: {val_loss[-1]:.6f} | "
        f"PSNR validación: {val_psnr[-1]:.4f} | "
        f"KL validación: {val_kl[-1]:.4f} | "
        f"μ latente: {val_latent_mean[-1]:.4f} | "
        f"σ latente: {val_latent_std[-1]:.4f}"
    )

    save_vae_images(
        model,
        val_loader,
        f"epoch_{epoch + 1:03d}.png",
    )
