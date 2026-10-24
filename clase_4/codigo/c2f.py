# %%
data_train_x = [
    -100.0,
    -95.0,
    -90.0,
    -85.0,
    -80.0,
    -75.0,
    -70.0,
    -65.0,
    -60.0,
    -55.0,
    -50.0,
    -45.0,
    -40.0,
    -35.0,
    -30.0,
    -25.0,
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
]
data_train_y = [
    -148.0,
    -139.0,
    -130.0,
    -121.0,
    -112.0,
    -103.0,
    -94.0,
    -85.0,
    -76.0,
    -67.0,
    -58.0,
    -49.0,
    -40.0,
    -31.0,
    -22.0,
    -13.0,
    -4.0,
    5.0,
    14.0,
    23.0,
    32.0,
    41.0,
    50.0,
    59.0,
    68.0,
    77.0,
    86.0,
    95.0,
    104.0,
    113.0,
    122.0,
    131.0,
    140.0,
    149.0,
    158.0,
    167.0,
    176.0,
    185.0,
    194.0,
    203.0,
    212.0,
]
data_val_x = [-97.5, -52.5, -2.5, 22.5, 47.5, 72.5, 97.5, -22.5, 12.5, 32.5]
data_val_y = [-143.5, -62.5, 27.5, 72.5, 117.5, 162.5, 207.5, -8.5, 54.5, 90.5]
# %%
import numpy as np

data_train_x = np.array(data_train_x)
data_train_y = np.array(data_train_y)
data_val_x = np.array(data_val_x)
data_val_y = np.array(data_val_y)

min_x = np.min(data_train_x)
max_x = np.max(data_train_x)
min_y = np.min(data_train_y)
max_y = np.max(data_train_y)

mean_x = np.mean(data_train_x)
std_x = np.std(data_train_x)
mean_y = np.mean(data_train_y)
std_y = np.std(data_train_y)

print(f"min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")
print(f"mean_x: {mean_x}, std_x: {std_x}, mean_y: {mean_y}, std_y: {std_y}")
# %%


def min_max_normalize(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


def min_max_denormalize(x_norm, min_x, max_x):
    return x_norm * (max_x - min_x) + min_x


def z_score_normalize(x, mean_x, std_x):
    return (x - mean_x) / std_x


def z_score_denormalize(x_norm, mean_x, std_x):
    return x_norm * std_x + mean_x


x_norm_mm = min_max_normalize(data_train_x, min_x, max_x)
y_norm_mm = min_max_normalize(data_train_y, min_y, max_y)
x_norm_zs = z_score_normalize(data_train_x, mean_x, std_x)
y_norm_zs = z_score_normalize(data_train_y, mean_y, std_y)

x_val_norm_mm = min_max_normalize(data_val_x, min_x, max_x)
y_val_norm_mm = min_max_normalize(data_val_y, min_y, max_y)

x_val_norm_zs = z_score_normalize(data_val_x, mean_x, std_x)
y_val_norm_zs = z_score_normalize(data_val_y, mean_y, std_y)

# min_max_normalize(5, min_x, max_x)
# z_score_normalize(5, mean_x, std_x)
# %%

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%

from torch.utils.data import Dataset, DataLoader


class C2FDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# sin normalización

# data_train = C2FDataset(data_train_x, data_train_y)
# data_val = C2FDataset(data_val_x, data_val_y)

# con normalización min-max

# data_train = C2FDataset(x_norm_mm, y_norm_mm)
# data_val = C2FDataset(x_val_norm_mm, y_val_norm_mm)

# con normalización z-score
data_train = C2FDataset(x_norm_zs, y_norm_zs)
data_val = C2FDataset(x_val_norm_zs, y_val_norm_zs)


# %%
import os


cpus = os.cpu_count()

data_train_loader = DataLoader(
    data_train, batch_size=32, shuffle=True, num_workers=cpus
)

data_val_loader = DataLoader(data_val, batch_size=64, shuffle=False, num_workers=cpus)


# %%
from torch import nn


class C2FModel(nn.Module):
    def __init__(self):
        super(C2FModel, self).__init__()
        self.fc1 = nn.Linear(1, 4, bias=True)
        self.fc2 = nn.Linear(4, 4, bias=True)
        self.fc3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# %%


def R_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# %%
from torch import optim


model = C2FModel().to(device)
lr = 1e-3
epochs = 10000
metricas_al = 50
loss_fn = nn.MSELoss()
mae = nn.L1Loss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)


# %%

for epoch in range(epochs):
    model.train()
    actual_loss = 0.0
    actual_mae = 0.0
    actual_r2 = 0.0
    # entrenamiento
    for x, y in data_train_loader:
        x = x.view(-1, 1).to(device)
        y = y.view(-1, 1).to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        loss.backward()

        optimizer.step()

        actual_loss += loss.item()
        actual_mae += mae(y_pred, y).item()
        actual_r2 += R_squared(y, y_pred).item()
    actual_loss /= len(data_train_loader)
    actual_mae /= len(data_train_loader)
    actual_r2 /= len(data_train_loader)
    if (epoch + 1) % metricas_al == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1}/{epochs},train values, Loss: {actual_loss:.4f}, MAE: {actual_mae:.4f}, R2: {actual_r2:.4f}"
        )

    model.eval()
    actual_loss = 0.0
    actual_mae = 0.0
    actual_r2 = 0.0

    # validación
    if (epoch + 1) % metricas_al == 0 or epoch == 0:
        with torch.no_grad():
            for x, y in data_val_loader:
                x = x.view(-1, 1).to(device)
                y = y.view(-1, 1).to(device)

                y_pred = model(x)

                loss = loss_fn(y_pred, y)

                actual_loss += loss.item()
                actual_mae += mae(y_pred, y).item()
                actual_r2 += R_squared(y, y_pred).item()

        print(
            f"Epoch {epoch + 1}/{epochs},val values, Loss: {actual_loss:.4f}, MAE: {actual_mae:.4f}, R2: {actual_r2:.4f}"
        )


# %%
