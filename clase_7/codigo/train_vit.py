# %%
import os
import time

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None


# ## Configuración

# %%
BATCH_SIZE = 64
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 100

if torch.cuda.is_available():
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
else:
    mixed_precision = "no"

accelerator = Accelerator(mixed_precision=mixed_precision)
device = accelerator.device
accelerator.print(f"Dispositivo: {device} | Precisión: {mixed_precision}")


# ## Dataset

# %%
dataset = load_dataset("cifar100")  # https://huggingface.co/datasets/uoft-cs/cifar100

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


class CIFARDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["img"]
        label = sample["fine_label"]

        if self.transform:
            image = self.transform(image)

        return image, label


train_data = CIFARDataset(dataset["train"], transform=train_transform)
val_data = CIFARDataset(dataset["test"], transform=test_transform)

num_workers = min(4, os.cpu_count() or 1)
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


# ## Vision Transformer


# %%
class Pachificador(nn.Module):
    def __init__(self, in_ch, emb_dim, patch_size):
        super(Pachificador, self).__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.seq(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_hidden_dim, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, mlp_hidden_dim, emb_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))


class VIT(nn.Module):
    def __init__(
        self,
        in_ch,
        emb_dim,
        patch_size,
        num_layers,
        num_heads,
        mlp_hidden_dim,
        w,
        h,
        dropout=0.1,
        class_num=10,
    ):
        super(VIT, self).__init__()
        self.pachificador = Pachificador(in_ch, emb_dim, patch_size)

        self.pos_emb = nn.Parameter(
            torch.randn(1, (w // patch_size) * (h // patch_size) + 1, emb_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        mod = [
            TransformerBlock(emb_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(num_layers)
        ]

        self.transformer_blocks = nn.Sequential(*mod)

        self.fc = nn.Linear(emb_dim, class_num)

    def forward(self, x):
        x = self.pachificador(x)
        batch_size, _, emb_dim = x.shape
        cls = self.cls_token.expand(batch_size, -1, emb_dim)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_emb
        x = self.transformer_blocks(x)
        x = x[:, 0]
        return self.fc(x)


# %%

# ## Entrenamiento y evaluación


def compute_metrics(logits, targets):
    loss = nn.functional.cross_entropy(logits, targets)

    preds = logits.argmax(dim=-1)
    accuracy = (preds == targets).float().mean()

    return loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with accelerator.autocast():
            logits = model(x)
            loss, acc = compute_metrics(logits, y)

        total_loss += loss.item()
        total_acc += acc.item()
        total_batches += 1

    mean_loss = total_loss / total_batches
    mean_acc = total_acc / total_batches

    return {
        "loss": mean_loss,
        "accuracy": mean_acc,
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for x, y in tqdm(
        loader,
        desc="train",
        leave=False,
        disable=not accelerator.is_local_main_process,
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with accelerator.autocast():
            logits = model(x)
            loss, acc = compute_metrics(logits, y)

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        total_batches += 1

    mean_loss = total_loss / total_batches
    mean_acc = total_acc / total_batches

    return {
        "loss": mean_loss,
        "accuracy": mean_acc,
    }


def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        elapsed = time.time() - start
        epoch_log = (
            f"Epoch {epoch}/{epochs} | "
            f"train_ce: {train_metrics['loss']:.4f} | "
            f"train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_ce: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"tiempo: {elapsed:.1f}s"
        )
        epoch_logs.append(epoch_log)
        epoch_logs = epoch_logs[-4:]

        if clear_output is not None and accelerator.is_main_process:
            clear_output(wait=True)
        elif accelerator.is_main_process:
            print("\033[2J\033[H", end="")

        accelerator.print("\n".join(epoch_logs))

    return history


# %%
model = VIT(
    in_ch=3,
    emb_dim=64,
    patch_size=PATCH_SIZE,
    num_layers=4,
    num_heads=4,
    mlp_hidden_dim=256,
    w=IMAGE_SIZE,
    h=IMAGE_SIZE,
    dropout=0.3,
    class_num=NUM_CLASSES,
)

# %%
from torch import optim

LEARNING_RATE = 1e-3
EPOCHS = 10
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    val_loader,
)

accelerator.print(
    f"Parámetros: {sum(parameter.numel() for parameter in model.parameters()):,}"
)

# %%
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
)
