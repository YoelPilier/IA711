# %%
from pathlib import Path
import os

import datasets
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %%
directorio_script = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
directorio_datos = directorio_script / "data"
directorio_datos.mkdir(parents=True, exist_ok=True)

output_dir = directorio_script / "resultados" / "gan"
output_dir.mkdir(parents=True, exist_ok=True)


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
dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
test_data = dataset["test"]

# %%
size = 64
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

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


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def Denormalize(tensor):
    return tensor * 0.5 + 0.5


def show_batch(batch, denormalize=None, filename=None):
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

    plt.tight_layout()
    if filename is not None:
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# %%
batch_size = 64
cpus = os.cpu_count()

train_dataset = CustomDataset(train_data, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=cpus,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

test_dataset = CustomDataset(test_data, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=cpus,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

x = next(iter(train_loader))
show_batch(x, denormalize=Denormalize, filename="datos.png")

# %%
import torch
from torch import nn
from torch.nn import functional as F


# %%
class ResidualCondBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualCondBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.shortcut = nn.Sequential()
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
        self.relu = nn.SiLU(inplace=True)

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(
            self.shortcut[0].weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(UpsampleBlock, self).__init__()
        self.conv = ResidualCondBlock(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# %%
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.fc = nn.Linear(latent_dim, 4 * 4 * 512)

        self.res_blocks = nn.Sequential(
            UpsampleBlock(
                512,
                256,
            ),
            UpsampleBlock(
                256,
                128,
            ),
            UpsampleBlock(
                128,
                64,
            ),
            UpsampleBlock(
                64,
                32,
            ),
        )

        self.conv_final = nn.Conv2d(
            32, img_channels, kernel_size=3, stride=1, padding=1
        )
        self.tanh = nn.Tanh()

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)
        nn.init.kaiming_normal_(
            self.conv_final.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv_final.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.res_blocks(x)
        x = self.conv_final(x)
        x = self.tanh(x)
        return x


# %%


class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.res_blocks = nn.Sequential(
            ResidualCondBlock(
                img_channels,
                64,
                stride=2,
            ),
            ResidualCondBlock(
                64,
                128,
                stride=2,
            ),
            ResidualCondBlock(
                128,
                256,
                stride=2,
            ),
            ResidualCondBlock(
                256,
                512,
                stride=2,
            ),
        )
        self.fc = nn.Linear(512, 1)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.flatten(x)
        return self.fc(x)


# %%
from accelerate import Accelerator

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

accelerator = Accelerator(mixed_precision="fp16")

lr_g = 2e-4
lr_d = 2e-4
epochs = 100
z_dim = 128
base_channels = 512

G = Generator(latent_dim=z_dim, img_channels=3)
D = Discriminator(img_channels=3)

G_ema = AveragedModel(
    G, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999), use_buffers=True
)

G_optimizer = torch.optim.AdamW(G.parameters(), lr=lr_g, betas=(0.0, 0.999))
D_optimizer = torch.optim.AdamW(D.parameters(), lr=lr_d, betas=(0.0, 0.999))

G, D, G_ema, G_optimizer, D_optimizer, train_loader, test_loader = accelerator.prepare(
    G, D, G_ema, G_optimizer, D_optimizer, train_loader, test_loader
)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    G.train()
    D.train()
    for real_imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        bs = real_imgs.size(0)
        real_imgs = real_imgs.to(accelerator.device)

        # Train Discriminator
        D_optimizer.zero_grad(set_to_none=True)
        z = torch.randn(bs, z_dim, device=accelerator.device)

        fake_imgs = G(z)
        real_preds = D(real_imgs)
        fake_preds = D(fake_imgs.detach())

        real_loss = loss_fn(real_preds, torch.ones_like(real_preds))
        fake_loss = loss_fn(fake_preds, torch.zeros_like(fake_preds))
        D_loss = (real_loss + fake_loss) / 2
        accelerator.backward(D_loss, retain_graph=True)
        D_optimizer.step()

        # Train Generator
        G_optimizer.zero_grad(set_to_none=True)
        z = torch.randn(bs, z_dim, device=accelerator.device)
        fake_imgs = G(z)
        fake_preds = D(fake_imgs)
        G_loss = loss_fn(fake_preds, torch.ones_like(fake_preds))
        accelerator.backward(G_loss)
        G_optimizer.step()
        G_ema.update_parameters(G)
    print(
        f"Epoch [{epoch + 1}/{epochs}] | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}"
    )
    with torch.no_grad():
        z = torch.randn(16, z_dim, device=accelerator.device)
        fake_imgs = G_ema(z)
    show_batch(
        fake_imgs,
        denormalize=Denormalize,
        filename=f"epoch_{epoch + 1:03d}.png",
    )


# %%
