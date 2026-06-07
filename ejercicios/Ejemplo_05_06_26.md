---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---
# Inteligencia Artificial

## Yoel Andeyci Pilier Martinez

### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---
# Ejemplo del 05/06/2026
## Datos
$$\text {x}=\begin{bmatrix}
-0.3536 & 0.8437 & 0.8841 \\
-0.3121 & 0.1697 & 0.3792 \\
\end{bmatrix}
\qquad
\text {y}=\begin{bmatrix}
2.2860 \\
-0.5254 \\
\end{bmatrix}$$

$$\text {w}=\begin{bmatrix}
-0.4148 \\
0.2671 \\
-1.2288 \\
\end{bmatrix}
\qquad
\text {b}=\begin{bmatrix}-0.2693\end{bmatrix}$$

---

# Predicción(Forward Pass)

$$\hat{y} =\text {x} \cdot \text {w}+\text {b}$$

$$\hat{y} =\begin{bmatrix}-0.3536 & 0.8437 & 0.8841 \\
-0.3121 & 0.1697 & 0.3792 \\
\end{bmatrix} \cdot \begin{bmatrix}
-0.4148 \\
0.2671 \\
-1.2288 \\
\end{bmatrix} + \begin{bmatrix} -0.2693 \end{bmatrix}$$

$$\hat{y} =\begin{bmatrix}-0.3536 \cdot -0.4148 + 0.8437 \cdot 0.2671 +
0.8841 \cdot -1.2288 -0.2693    \\ 
-0.3121 \cdot -0.4148 + 0.1697 \cdot 0.2671 + 0.3792 \cdot -1.2288 -0.2693 \\
\end{bmatrix}$$


$$\hat{y} =\begin{bmatrix}-0.9837 \\
-0.5605 \\
\end{bmatrix}$$

---

# Cálculo del error
## Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i )^2$$

$$\text{MSE} = \frac{1}{2} [(-0.9837 - 2.2860)^2 + (-0.5605 + 0.5254)^2]$$

$$\text{MSE} = \frac{1}{2} [(-3.2697)^2 + (-0.0351)^2]$$

$$\text{MSE} = \frac{1}{2} [10.6910 + 0.0013]$$

$$\text{MSE} = \frac{1}{2} [10.6923]$$

$$\text{MSE} \approx 5.3461$$

---
# Backpropagation

## Gradiente del error con respecto a las predicciones(derivada de la función de pérdida)

$$\text{MSE}'(\hat{y}) = \frac{1}{n} 2(\hat{y}_i - y_i)$$

$$\text{MSE}'(\hat{y}) = \frac{1}{2} 2(\hat{y}_i - y_i)$$

$$\text{MSE}'(\hat{y}) = \hat{y}_i - y_i$$

$$\text{MSE}'(\hat{y}) = \begin{bmatrix}
-0.9837 - 2.2860 \\ 
-0.5605 + 0.5254 \\
\end{bmatrix}$$

$$\text{MSE}'(\hat{y}) = \begin{bmatrix}
-3.2697 \\ 
-0.0351 \\
\end{bmatrix}$$

---

## Backpropagation
# Gradiente del error con respecto al sesgo

$$\nabla{b} = \sum \text{MSE}' $$

$$\nabla{b} = -3.2697 + -0.0351 $$

$$\nabla{b} = -3.3048   $$


---

# Backpropagation
## Gradiente del error con respecto a los pesos 

$$\nabla{w} = \text {x}^T \cdot \text{MSE}'$$

$$\nabla{w} = \begin{bmatrix}
-0.3536 & -0.3121 \\
0.8437 & 0.1697 \\
0.8841 & 0.3792 \\
\end{bmatrix} \cdot 
\begin{bmatrix}
-3.2697 \\ 
-0.0351 \\
\end{bmatrix}$$

$$\nabla{w} = \begin{bmatrix}
-0.3536 \cdot -3.2697 + -0.3121 \cdot -0.0351 \\
0.8437 \cdot -3.2697 + 0.1697 \cdot -0.0351 \\
0.8841 \cdot -3.2697 + 0.3792 \cdot -0.0351 \\
\end{bmatrix}$$

$$\nabla{w} = \begin{bmatrix}
1.1671 \\ 
-2.7647 \\ 
-2.9041 \\
\end{bmatrix}$$

---

# Stochastic Gradient Descent (SGD)

## Actualización de los pesos y el sesgo

$$\text{w} = \text{w} - \alpha \nabla{w}$$

$$\text{b} = \text{b} - \alpha \nabla{b}$$

Donde $$\alpha$$ es el Learning rate (tasa de aprendizaje).

$$\alpha = 0.01$$

$$\text{w} = \begin{bmatrix}
-0.4148 \\ 
0.2671 \\ 
-1.2288 \\
\end{bmatrix} - 0.01 \cdot 
\begin{bmatrix}1.1671 \\
-2.7647 \\
-2.9041 \\
\end{bmatrix}$$

$$\text{b} = \begin{bmatrix}
-0.2693
\end{bmatrix} - 0.01 \cdot -3.3048$$


---

# Resultados después de la actualización

$$\text{w} = \begin{bmatrix}
-0.4265 \\
0.2948 \\ 
-1.1998 \\
\end{bmatrix}$$

$$\text{b} = \begin{bmatrix}
-0.2363
\end{bmatrix}$$

---

# Predicción después de la actualización

$$\hat{y} =\text {x} \cdot \text {w}+\text {b}$$

$$\hat{y} =\begin{bmatrix}
-0.3536 & 0.8437 & 0.8841 \\
-0.3121 & 0.1697 & 0.3792 \\
\end{bmatrix} \cdot 
\begin{bmatrix}
-0.4265 \\ 0.2948 \\ -1.1998 \\
\end{bmatrix}+
\begin{bmatrix}
-0.2363
\end{bmatrix}$$

$$\hat{y} =\begin{bmatrix}
-0.3536 \cdot -0.4265 + 0.8437 \cdot 0.2948 + 0.8841 \cdot -1.1998 -0.2363\\
-0.3121 \cdot -0.4265 + 0.1697 \cdot 0.2948 + 0.3792 \cdot -1.1998 -0.2363\\
\end{bmatrix}$$

$$\hat{y} =\begin{bmatrix}
-0.8976 \\
-0.5082 \\
\end{bmatrix}$$

---
# Cálculo del error después de la actualización

$$\text{MSE} = \frac{1}{2} [(-0.8937 - 2.2860)^2 + (-0.5082 - 0.5254)^2]$$

$$\text{MSE} = \frac{1}{2} [(-3.1797)^2 + (0.0172)^2]$$

$$\text{MSE} = \frac{1}{2} [10.1105 + 0.0003]$$

$$\text{MSE} = \frac{1}{2} [10.1108]$$

$$\text{MSE} \approx 5.0554$$

---

# Mejora del error

$$\text{Mejora} = \frac{\text{Error anterior} - \text{Error actual}}{\text{Error anterior}} \times 100\%$$

$$\text{Mejora} = \frac{5.3461 - 5.0554}{5.3461} \times 100\%$$

$$\text{Mejora} \approx 5.44\%$$

---

# Version en Python

```Python
import torch

# Datos

x = torch.tensor([
    [-0.3536, 0.8437, 0.8841],
    [-0.3121, 0.1697, 0.3792]
])

y = torch.tensor([
    [2.2860],
    [-0.5254]
])

w = torch.tensor([
    [-0.4148],
    [0.2671],
    [-1.2288]
])

b = torch.tensor([[-0.2693]])

print("Parámetros iniciales")
print("w:")
print(w)
print("\nb:")
print(b)

# Forward pass

y_hat = x @ w + b

print("\nPredicción inicial")
print(y_hat)

# Pérdida inicial

loss = torch.mean((y_hat - y) ** 2)

print("\nPérdida inicial (MSE)")
print(loss.item())

# Backward pass manual

d_loss = (2 / y.numel()) * (y_hat - y)

d_b = d_loss.sum(dim=0)
d_w = x.T @ d_loss

print("\nGradientes")
print("d_w:")
print(d_w)

print("\nd_b:")
print(d_b)

# Optimización

lr = 0.01

w -= lr * d_w
b -= lr * d_b

print("\nParámetros actualizados")
print("w:")
print(w)

print("\nb:")
print(b)

# Segunda predicción

y_hat = x @ w + b

print("\nPredicción después de la actualización")
print(y_hat)

# Segunda pérdida

loss_2 = torch.mean((y_hat - y) ** 2)

print("\nPérdida después de la actualización (MSE)")
print(loss_2.item())

# Mejora

mejora = (loss - loss_2) / loss * 100

print("\nComparación")
print(f"MSE inicial : {loss.item():.6f}")
print(f"MSE final   : {loss_2.item():.6f}")
print(f"Mejora      : {mejora.item():.4f}%")

```

