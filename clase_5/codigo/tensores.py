# %%
import torch

t1 = torch.randn(2, 3, 24, 24)
t1_flat = t1.reshape(2, 3 * 24 * 24)
l1 = torch.nn.Linear(3 * 24 * 24, 10)
out = l1(t1_flat)
out

# %%
