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
# Ejercicios

Para que un ejercicio se considere resuelto, debe realizar los siguientes pasos con lapiz y papel, y luego implementar el código en Python:

1. Realizar la predicción (*forward pass*).
2. Calcular el error (*loss*) utilizando MSE (*Mean Squared Error*) o MAE (*Mean Absolute Error*).
3. Calcular las derivadas (*backward pass*).
4. Actualizar los pesos y el sesgo (*weight and bias update*).
5. Realizar una nueva predicción (*forward pass*) con los pesos y el sesgo actualizados.
6. Comparar el error antes y después de la actualización.

---


## Información necesaria

### Funciones de pérdida

MSE (*Mean Squared Error*):

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

MAE (*Mean Absolute Error*):

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|
$$


donde:

- (N) es la cantidad de muestras.
- ($$\hat y$$) es la predicción del modelo.
- (y) es el valor objetivo.

---

### Derivadas de las funciones de pérdida

- MSE:

$$
\text{MSE}'=\frac{2}{N}(\hat y_i-y_i)
$$

- MAE derivada

$$\text{MAE}' = \frac{1}{n}\cdot\text{sign}(\hat{y} - y)$$

donde:

$$
\text{sign}(x)=
\begin{cases}
+1 & x > 0 \\
0 & x = 0 \\
-1 & x < 0
\end{cases}
$$

---
### Funciones de activación

- ReLU:

$$
f(x)=\max(0,x)
$$

- Sigmoid:

$$
f(x)=\frac{1}{1+\exp(-x)}
$$

donde (\exp) es la función exponencial, es decir, elevar el número (e) a la potencia (-x). El número (e) es la constante de Euler y es aproximadamente igual a 2.71828.

Por lo tanto, la función sigmoide también puede calcularse como:

$$
f(x)=\frac{1}{1+2.71828^{-x}}
$$

- Tanh:

$$
f(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
$$

---

### Derivadas de las funciones de activación

- ReLU:

$$
f'(x)=
\begin{cases}
1 & \text{si } x > 0\
0 & \text{si } x \le 0
\end{cases}
$$

- Sigmoid:

$$
f'(x)=f(x)\cdot(1-f(x))
$$

donde (f(x)) es la salida de la función sigmoide.

- Tanh:

$$
f'(x)=1-f(x)^2
$$


---

## Como calcular los gradientes con mas de 1 capa

El secreto esta en como calculamos los gradientes de los pesos. Usamos la entrada de la capa transpuesta multiplicada por el gradiente que viene de las capas siguientes. Entonces, lo que vamos a pasar a la capa anterior es el peso de esta capa multiplicado por el gradiente que viene de las capas siguientes y por la derivada de la funcion de activacion. Asi sucesivamente hasta llegar a la primera capa.


---
## Ejercicio 1

$$
\hat{y} = \text{sigmoid}\big(\text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = MSE**

$$
x=
\begin{bmatrix}
0.5636 & 0.7666 \\
0.1621 & 0.4935 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.9233 \\
0.3352 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
1.0 & -0.5 \\
0.3 & 0.8 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
-0.2 & 0.1 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.7 \\
-0.4 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.05 \\
\end{bmatrix}
$$

---

## Ejercicio 2

$$
\hat{y} = \text{relu}\big(\text{sigmoid}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.005**

**loss = MAE**

$$
x=
\begin{bmatrix}
-0.3536 & -0.3121 \\
0.8437 & 0.1697 \\
0.8841 & 0.3792 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
-0.9837 \\
-0.5605 \\
0.5254 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
-0.2 & 0.6 \\
0.4 & -0.1 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.0 & -0.05 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.3 \\
-0.7 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
-0.1 \\
\end{bmatrix}
$$

---

## Ejercicio 3

$$
\hat{y} = \text{tanh}\big(\text{relu}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.02**

**loss = MSE**

$$
x=
\begin{bmatrix}
0.5636 & 0.7666 \\
0.1621 & 0.4935 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.2 \\
-0.4 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
1.2 & -0.8 \\
0.5 & 0.3 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.1 & 0.2 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.6 \\
-0.9 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.0 \\
\end{bmatrix}
$$

---

## Ejercicio 4

$$
\hat{y} = \text{sigmoid}\big(\text{sigmoid}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = MAE**

$$
x=
\begin{bmatrix}
-0.3536 & -0.3121 & 0.8437 \\
0.1697 & 0.8841 & 0.3792 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.9837 \\
0.5605 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
-0.3 & 0.2 & 0.5 \\
0.1 & -0.4 & 0.7 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
-0.05 & 0.05 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.4 \\
-0.6 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.02 \\
\end{bmatrix}
$$

---

## Ejercicio 5

$$
\hat{y} = \text{relu}\big(\text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = MSE**

$$
x=
\begin{bmatrix}
0.2 & 0.5 & 0.8 \\
0.7 & 0.1 & 0.4 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.4 \\
0.8 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.3 & -0.2 \\
0.7 & 0.4 \\
-0.5 & 0.8 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.1 & -0.1 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.2 \\
-0.3 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.05 \\
\end{bmatrix}
$$

---

## Ejercicio 6

$$
\hat{y} = \text{tanh}\big(\text{sigmoid}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.05**

**loss = MAE**

$$
x=
\begin{bmatrix}
0.3 & 0.7 \\
0.8 & 0.2 \\
0.5 & 0.9 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
1 \\
0 \\
1 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.6 & -0.4 \\
0.2 & 0.9 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.1 & -0.2 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.5 \\
-0.1 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.0 \\
\end{bmatrix}
$$

---

## Ejercicio 7

$$
\hat{y} = \text{sigmoid}\big(\text{relu}( \text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}) \cdot W^{(3)} + b^{(3)}\big)
$$

**lr = 0.01**

**loss = MSE**

$$
x=
\begin{bmatrix}
0.1 & 0.5 & 0.7 \\
0.3 & 0.9 & 0.2 \\
0.8 & 0.1 & 0.4  \\
0.6 & 0.7 & 0.3 \\ 
0.2 & 0.4 & 0.9 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.8 \\
1.1 \\
0.3 \\
0.9 \\
0.5 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.2 & -0.1 \\
0.4 & 0.3 \\
-0.2 & 0.5 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.0 & 0.05 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.6 & -0.3 \\
0.1 & 0.4 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
-0.02 & 0.02 \\
\end{bmatrix}
$$

$$
W^{(3)}=
\begin{bmatrix}
0.4 \\
-0.2 \\
\end{bmatrix}
\qquad
b^{(3)}=
\begin{bmatrix}
0.1 \\
\end{bmatrix}
$$

---

## Ejercicio 8

$$
\hat{y} = \text{tanh}\big(\text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.02**

**loss = MAE**

$$
x=
\begin{bmatrix}
0.45 & -0.12 \\
-0.33 & 0.88 \\
0.77 & 0.05 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.1 \\
-0.2 \\
0.5 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.5 & -0.6 \\
0.2 & 0.3 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.0 & 0.1 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
-0.4 \\
0.9 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
-0.05 \\
\end{bmatrix}
$$

---

## Ejercicio 9

$$
\hat{y} = \text{relu}\big(\text{sigmoid}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.005**

**loss = MSE**

$$
x=
\begin{bmatrix}
0.12 & 0.34 & 0.56 \\
0.78 & 0.90 & 0.11 \\
0.22 & 0.44 & 0.66 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.3 \\
0.7 \\
0.2 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.1 & -0.2 \\
0.3 & 0.4 \\
-0.1 & 0.2 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.01 & -0.01 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.25 \\
-0.35 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.0 \\
\end{bmatrix}
$$

---

## Ejercicio 10

$$
\hat{y} = \text{sigmoid}\big(\text{relu}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = MAE**

$$
x=
\begin{bmatrix}
0.33 & 0.66 \\
0.11 & 0.22 \\
0.77 & 0.88 \\
0.44 & 0.55 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
1 \\
0 \\
1 \\
0 \\
\end{bmatrix}
$$

$$
W^{(1)}=
\begin{bmatrix}
0.6 & -0.3 \\
0.2 & 0.5 \\
\end{bmatrix}
\qquad
b^{(1)}=
\begin{bmatrix}
0.05 & -0.05 \\
\end{bmatrix}
$$

$$
W^{(2)}=
\begin{bmatrix}
0.45 \\
-0.25 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.02 \\
\end{bmatrix}
$$
