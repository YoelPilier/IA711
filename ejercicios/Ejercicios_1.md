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
2. Calcular el error (*loss*) utilizando MSE (*Mean Squared Error*).
3. Calcular las derivadas (*backward pass*).
4. Actualizar los pesos y el sesgo (*weight and bias update*).
5. Realizar una nueva predicción (*forward pass*) con los pesos y el sesgo actualizados.
6. Comparar el error antes y después de la actualización.

---


## Información necesaria

### Función de pérdida

MSE (*Mean Squared Error*):

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

donde:

- (N) es la cantidad de muestras.
- (\hat y) es la predicción del modelo.
- (y) es el valor objetivo.

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

---

## Ejercicio 1

$
\hat{y} = x \cdot w + b
$

**lr = 0.001**

$
x=
\begin{bmatrix}
0.5636 & 0.7666 \\
0.1621 & 0.4935 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
-0.9233 \\
-1.3352 \\
\end{bmatrix}
$

$
w=
\begin{bmatrix}
1.3696 \\
-0.0677 \\ 
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
-1.6129 \\
\end{bmatrix}
$

---

## Ejercicio 2

$
\hat{y} = x \cdot w + b
$

**lr = 0.01**

$
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
$

$
w=
\begin{bmatrix}
-0.3536 \\
-0.3121 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
-0.3121 \\
\end{bmatrix}
$

---

## Ejercicio 3

$
\hat{y} = \text{sigmoid}(x \cdot w + b)
$

**lr = 0.1**

$
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
$

$
w=
\begin{bmatrix}
1.3696 \\
-0.0677 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
-1.6129 \\
\end{bmatrix}
$

---

## Ejercicio 4

$
\hat{y} = \text{relu}(x \cdot w + b)
$

**lr = 0.01**

$
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
$

$
w=
\begin{bmatrix}
-0.3536 \\
-0.3121 \\
0.8437 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
-0.3121 \\
\end{bmatrix}
$

---

## Ejercicio 5

$
\hat{y} = \text{sigmoid}(x \cdot w + b)
$

**lr = 0.01**

$
x=
\begin{bmatrix}
0.2 & 0.5 & 0.8 \\
0.7 & 0.1 & 0.4 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.4 & 0.9 \\
0.8 & 0.3 \\
\end{bmatrix}
$

$
w=
\begin{bmatrix}
0.3 & -0.2 \\
0.7 & 0.4 \\
-0.5 & 0.8 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.1 & -0.1 \\
\end{bmatrix}
$

---

## Ejercicio 6

$
\hat{y}=\text{sigmoid}(x \cdot w +b)
$

**lr = 0.05**

$
x=
\begin{bmatrix}
0.3 & 0.7 \\
0.8 & 0.2 \\
0.5 & 0.9 \\
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1 \\
\end{bmatrix}
$

$
W=
\begin{bmatrix}
0.6 & -0.4 \\
0.2 & 0.9 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.1 & -0.2 \\
\end{bmatrix}
$

---

## Ejercicio 7

$
\hat{y}=\text{ReLU}(x \cdot w +b)
$


**lr = 0.01**

$
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
$

$
w=
\begin{bmatrix}
0.4 \\
-0.2 \\
0.7 \\
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.1 \\
\end{bmatrix}
$

