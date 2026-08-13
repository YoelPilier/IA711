# %%
import subprocess
import sys
import os
from pathlib import Path


from datasets import load_dataset

# ## Dataset

numero_articulos = 20_000
ruta = Path("/tmp/data/wikipedia_es/wikipedia_es.txt")
ruta.parent.mkdir(parents=True, exist_ok=True)

if not ruta.exists():
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.es",
        split="train",
        streaming=True,
    )
    with ruta.open("w", encoding="utf-8") as archivo:
        for indice, articulo in enumerate(dataset):
            if indice >= numero_articulos:
                break
            texto_articulo = articulo["text"].replace("\n", " ").strip()
            if texto_articulo:
                archivo.write(texto_articulo + "\n")


# ## Tokenización

# %%
# imports
import math
import sentencepiece as spm
import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

if torch.cuda.is_available():
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
else:
    mixed_precision = "no"

accelerator = Accelerator(mixed_precision=mixed_precision)
accelerator.print(f"Dispositivo: {accelerator.device} | Precisión: {mixed_precision}")

directorio_script = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
output_dir = directorio_script / "resultados" / "causal_transformer_wikipedia_es"
output_dir.mkdir(parents=True, exist_ok=True)
prompt_generacion = "La inteligencia artificial"


# %%

vocab_size = 12000
spm.SentencePieceTrainer.train(
    input=str(ruta),
    model_prefix="tokenizer",
    vocab_size=vocab_size,
    model_type="unigram",
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    unk_id=0,
)

# %%

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")


def encode(text, add_special_tokens=False):
    ids = sp.encode_as_ids(text)
    if add_special_tokens:
        ids = [sp.bos_id()] + ids + [sp.eos_id()]
    return torch.tensor(ids, dtype=torch.long)


def decode(tokens):
    return sp.decode(tokens.tolist())


# %%

tokens = encode("En el principio creó Dios", add_special_tokens=True)
print(tokens)
print(decode(tokens))


# %%

with open(ruta, "r") as f:
    text = f.read().splitlines()
print(f"Dataset has {len(text)} lines")
print(f"Example line: {text[0]}")


# %%

eos = torch.tensor([sp.eos_id()], dtype=torch.long)
tokens_planos = torch.cat(
    [torch.cat([encode(line), eos]) for line in text if line.strip()]
)

print(f"Corpus has {len(tokens_planos)} tokens")
print(f"Example tokens: {tokens_planos[:20]}")

# %%
split = int(0.9 * len(tokens_planos))
train_tokens = tokens_planos[:split]
val_tokens = tokens_planos[split:]


class TextDataset(Dataset):
    def __init__(self, tokens, contexto):
        self.tokens = tokens
        self.contexto = contexto
        self.block_size = contexto + 1
        self.num_windows = max(0, (len(tokens) - self.block_size) // contexto + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.contexto
        chunk = self.tokens[start : start + self.block_size]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


ventana_de_contexto = 128

train_data = TextDataset(train_tokens, ventana_de_contexto)
val_data = TextDataset(val_tokens, ventana_de_contexto)

print(f"Train windows: {len(train_data)}")
print(f"Val windows: {len(val_data)}")
print(train_data[0])


# %%
cpu_count = os.cpu_count()
batch_size = 64

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=cpu_count,
    pin_memory=True,
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=cpu_count,
    pin_memory=True,
)

x, y = next(iter(train_loader))
print(f"Batch x shape: {x.shape}")
print(f"Batch y shape: {y.shape}")

# %%


def param_count(model, as_string=True):
    n = sum(p.numel() for p in model.parameters())
    if not as_string:
        return n
    for unit, div in (("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pos = torch.arange(max_len)[:, None]

        i = torch.arange(0, d_model, 2)

        freq = 1 / (10000 ** (i / d_model))

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)

        self.register_buffer("pe", pe[None])

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=5000, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1 / self.d_k**0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))

        self.register_buffer("mask", mask[None, None])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=False):

        B, T, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores * self.scale

        if causal:
            mask = self.mask[:, :, :T, :T]

            scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        heads = attention @ V

        heads = heads.transpose(1, 2).reshape(B, T, D)

        return self.W_o(heads)


# %%


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# %%


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, max_len=max_len
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal=False):

        attn_out = self.attention(x, causal=causal)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# %%


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout=dropout, max_len=max_len
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        self.output_layer.weight = self.token_embedding.weight

    def forward(self, x):

        x = self.token_embedding(x)
        x = x * self.d_model**0.5
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, causal=True)

        logits = self.output_layer(x)

        return logits


# %%


def compute_metrics(logits, targets):
    b, t, vocab = logits.shape
    logits_flat = logits.view(b * t, vocab)
    targets_flat = targets.view(b * t)

    loss = F.cross_entropy(logits_flat, targets_flat)

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
    perplexity = math.exp(mean_loss)

    return {
        "loss": mean_loss,
        "accuracy": mean_acc,
        "perplexity": perplexity,
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
        "val_ppl": [],
    }
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_ppl"].append(val_metrics["perplexity"])

        epoch_log = (
            f"Epoch {epoch}/{epochs} | "
            f"train_ce: {train_metrics['loss']:.4f} | "
            f"train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_ce: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"val_ppl: {val_metrics['perplexity']:.4f}"
        )
        epoch_logs.append(epoch_log)
        epoch_logs = epoch_logs[-4:]

        if clear_output is not None and accelerator.is_main_process:
            clear_output(wait=True)
        elif accelerator.is_main_process:
            print("\033[2J\033[H", end="")

        accelerator.print("\n".join(epoch_logs))

        if accelerator.is_main_process:
            texto = generar(
                accelerator.unwrap_model(model),
                prompt=prompt_generacion,
                max_new_tokens=80,
                temperature=0.6,
                top_k=50,
            )
            (output_dir / f"epoch_{epoch:03d}.txt").write_text(
                f"{epoch_log}\n\n{texto}\n",
                encoding="utf-8",
            )

    return history


@torch.inference_mode()
def generar(
    model,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
):
    model.eval()

    if isinstance(prompt, str):
        idx = encode(prompt).unsqueeze(0).to(device)
    else:
        idx = prompt.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ventana_de_contexto:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, k=top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == sp.eos_id():
            break

        idx = torch.cat([idx, next_token], dim=1)

    return decode(idx[0].cpu())


# %%
from torch import optim

device = accelerator.device

dim = 512
num_heads = 8
num_layers = 6
dropout = 0.2

model = Transformer(
    vocab_size=vocab_size,
    d_model=dim,
    n_heads=num_heads,
    d_ff=dim * 4,
    n_layers=num_layers,
    dropout=dropout,
).to(device)


print(f"Model has {param_count(model)} parameters")

texto = generar(
    model,
    prompt=prompt_generacion,
    max_new_tokens=80,
    temperature=0.6,
    top_k=50,
)

print(texto)

# %%
lr = 1e-3
epochs = 100

optimizer = optim.AdamW(model.parameters(), lr=lr)
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    val_loader,
)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=epochs,
)

# %%
texto = generar(
    model,
    prompt=prompt_generacion,
    max_new_tokens=80,
    temperature=0.6,
    top_k=50,
)
print(texto)


# %%
