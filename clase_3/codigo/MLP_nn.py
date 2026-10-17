# %%
import torch
from torch import nn

lr = 0.1
momentum = 0.9
epochs = 10000

x = torch.tensor([[0.2, -1], [0, 0.5], [0.3, 0]])

y = torch.tensor([[0.3], [0.5], [0]])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 2, bias=True)
        self.w2 = nn.Linear(2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.w1(x)
        a1 = self.sigmoid(z1)
        y_hat = self.w2(a1)
        return y_hat


model = MLP()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

# %%
for epoch in range(epochs):
    optimizer.zero_grad()
    # forward pass
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    # %%
    # backward pass
    loss.backward()
    # step
    optimizer.step()

y_hat = model(x)

print(f"loss {loss_fn(y_hat, y).item()}")
