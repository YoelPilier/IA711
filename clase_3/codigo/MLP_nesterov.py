# %%
import torch

lr = 0.1
momentum = 0.9
epochs = 5000

x = torch.tensor([[0.2, -1], [0, 0.5], [0.3, 0]])

y = torch.tensor([[0.3], [0.5], [0]])

w1 = torch.tensor([[1, 0.3], [2, 0.1]])
b1 = torch.zeros(2)
w2 = torch.tensor([[0.2], [0.4]])

mw1 = torch.zeros_like(w1)
mb1 = torch.zeros_like(b1)
mw2 = torch.zeros_like(w2)


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
for epoch in range(epochs):
    # forward pass
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    y_hat = a1 @ w2
    loss = MSE(y_hat, y)

    # %%
    # backward pass

    dloss = dMSE(y_hat, y)
    dw2 = a1.T @ dloss
    da1 = dloss @ w2.T
    dz1 = da1 * dsigmoid(z1)
    db1 = dz1.sum(dim=0)
    dw1 = x.T @ dz1

    # step
    mw1 = mw1 * momentum + dw1
    mb1 = mb1 * momentum + db1
    mw2 = mw2 * momentum + dw2

    uw1 = dw1 + momentum * mw1
    ub1 = db1 + momentum * mb1
    uw2 = dw2 + momentum * mw2

    w1 = w1 - lr * uw1
    b1 = b1 - lr * ub1
    w2 = w2 - lr * uw2

z1 = x @ w1 + b1
a1 = sigmoid(z1)
y_hat = a1 @ w2
loss = MSE(y_hat, y)
print(f"Loss {loss}")
# %%
