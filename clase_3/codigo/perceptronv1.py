# %%

import torch

# %%
x = torch.tensor([[0.2, 3.0, 1.0], [1.0, 4.0, 6.0]])

y = torch.tensor([[0.2, 0.3], [0.2, 0.4]])

w = torch.tensor([[0.0, 4.0], [1.0, 6.0], [3.0, 2.0]])

b = torch.tensor([0.0, 0.0])


# %%


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def d_sigmoid(x):
    res = sigmoid(x)
    return res * (1 - res)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def d_mse_loss(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.numel()


# %%
xw = x @ w  # matrix multiplication

xwb = xw + b  # broadcasting

y_hat = sigmoid(xwb)

loss = mse_loss(y_hat, y)

loss

# %%
d_loss = d_mse_loss(y_hat, y)
d_sig = d_sigmoid(xwb)
d_z = d_loss * d_sig
d_d = d_z.sum(dim=0)
d_w = x.T @ d_z
d_w
d_x = d_z @ w.T

# %%
learning_rate = 1e-3

w -= learning_rate * d_w
b -= learning_rate * d_d

print(w)
print(b)
# %%
