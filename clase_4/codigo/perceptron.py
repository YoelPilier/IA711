# %%
import torch
import torch.nn as nn
from torch.optim import SGD

# %%

x = torch.tensor([[0.2, 3.0, 1.0], [1.0, 4.0, 6.0]])

y = torch.tensor([[0.2, 0.3], [0.2, 0.4]])


w = nn.Linear(3, 2, bias=True)

# %%


sigmoid = nn.Sigmoid()
mse_loss = nn.MSELoss()

y_hat = 0


learning_rate = 1e-1  # 1e-2, 1e-3, 1e-4
momentum = 0.9
epocas = 100000


optimizer = SGD([w.weight, w.bias], lr=learning_rate, momentum=momentum, nesterov=True)

# %%
for i in range(epocas):
    optimizer.zero_grad()

    y_hat = sigmoid(w(x))

    loss = mse_loss(y_hat, y)

    loss.backward()

    optimizer.step()

    if i % 1000 == 0:
        print(f"epoca: {i} perdida: {loss}")
    # %%


# %%
print("Predicted y:")
print(y_hat)
print("Original y:")
print(y)
