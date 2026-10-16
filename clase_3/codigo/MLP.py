# %%
import torch

lr = 0.1

x = torch.tensor([[0.2, -1], [0, 0.5], [0.3, 0]])

y = torch.tensor([[0.3], [0.5], [0]])

w1 = torch.tensor([[1, 0.3], [2, 0.1]])
b1 = torch.zeros(2)

w2 = torch.tensor([[0.2], [0.4]])


# %%
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def MSE(y_hat, y):
    return torch.mean((y_hat - y) ** 2)


def dsigmoid(x):
    delta = sigmoid(x)
    return delta * (1 - delta)


def dMSE(y_hat, y):
    return 2 * (y_hat - y) / y.size(0)


# %%
# forward pass
z1 = x @ w1 + b1
a1 = sigmoid(z1)
y_hat = a1 @ w2
loss = MSE(y_hat, y)
print(f"Loss {loss}")

# %%
# backward pass

dloss = dMSE(y_hat, y)
dw2 = a1.T @ dloss
da1 = dloss @ w2.T
dz1 = da1 * dsigmoid(z1)
db1 = dz1.sum(dim=0)
dw1 = x.T @ dz1

# step
w1 = w1 - lr * dw1
b1 = b1 - lr * db1
w2 = w2 - lr * dw2

# %%
