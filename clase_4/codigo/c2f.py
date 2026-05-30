# %%

import numpy as np

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
data_test_x = [-100.0, -50.0, 0.0, 25.0, 50.0, 75.0, 100.0, -20.0, 10.0, 30.0]
data_test_y = [-148.0, -58.0, 32.0, 77.0, 122.0, 167.0, 212.0, -4.0, 50.0, 86.0]

min_x = min(data_train_x)
max_x = max(data_train_x)
min_y = min(data_train_y)
max_y = max(data_train_y)


def normalizar(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


def denormalizar(y, min_y, max_y):
    return y * (max_y - min_y) + min_y


normalized_train_x = [normalizar(x, min_x, max_x) for x in data_train_x]
normalized_train_y = [normalizar(y, min_y, max_y) for y in data_train_y]

normalized_test_x = [normalizar(x, min_x, max_x) for x in data_test_x]
normalized_test_y = [normalizar(y, min_y, max_y) for y in data_test_y]

# %%
from torch.utils.data import Dataset, DataLoader


class Tdataset(Dataset):
    def __init__(self, x, y):
        super(Tdataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# train_dataset = Tdataset(normalized_train_x, normalized_train_y)
# test_dataset = Tdataset(normalized_test_x, normalized_test_y)
train_dataset = Tdataset(data_train_x, data_train_y)
test_dataset = Tdataset(data_test_x, data_test_y)

bs = 10

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# %%
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 4, bias=True)
        self.linear2 = nn.Linear(4, 4, bias=True)
        self.fc = nn.Linear(4, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.fc(x)
        return x


# Model()(torch.tensor([[0.0]]).float()).shape
# %%
lr = 1e-3
epochs = 10000


loss_fn = nn.MSELoss()
model = Model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

for epoch in range(epochs):
    model.train()
    for x, y in train_dataloader:
        optimizer.zero_grad()
        x = x.float().unsqueeze(1).to(device)
        y = y.float().unsqueeze(1).to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x, y in test_dataloader:
            x = x.float().unsqueeze(1).to(device)
            y = y.float().unsqueeze(1).to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_dataloader)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Test Loss: {avg_loss}")
# %%
