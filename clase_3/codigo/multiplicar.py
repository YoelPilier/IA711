# %%

x = 5
y = 10
lr = 1e-2
w = 0.3
b = 0

delta_w = 0
delta_b = 0
# %%

# predict

y_hat = x * w + b

# perdida
loss = (y_hat - y) ** 2

# backprop
dloss = 2 * (y_hat - y)
delta_b = dloss * 1
delta_w = dloss * x

# optimization

b -= lr * delta_b
w -= lr * delta_w

delta_w = 0
delta_b = 0

print(f"y_hat {y_hat} loss {loss}")
print(f"w {w} delta_w {delta_w} b {b} delta_b {delta_b}")

# %%

10000 * w + b

# %%
