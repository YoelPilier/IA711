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

## 1 Usando los scripts de clasificación de la clase,resuelve estos problemas de clasificación.

- zalando-datasets/fashion_mnist
- tanganke/kmnist

## 2 Para que uno de los siguientes ejercicios se considere resuelto, debe realizar los siguientes pasos con lapiz y papel, y luego implementar el código en Python:

1. Realizar la predicción (forward pass).
2. Calcular el error (loss) utilizando la función de pérdida indicada (ahora de clasificación: BCELoss o CrossEntropyLoss).
3. Calcular las derivadas (backward pass).
4. Actualizar los pesos y el sesgo (weight and bias update).
5. Realizar una nueva predicción (forward pass) con los pesos y el sesgo actualizados.
6. Comparar el error antes y después de la actualización.
7. Calcular la accuracy correspondiente (para BCELoss: accuracy binaria; para CrossEntropyLoss: accuracy multiclase).

---

## Información necesaria

### Funciones de pérdida (clasificación)

**BCELoss (Binary Cross Entropy Loss)**

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left(
y_i \log(p_i)
+
(1-y_i)\log(1-p_i)
\right)
$$

$$ P_i = \frac{1}{1 + e^{-z_i}} $$

Derivada:

$$ \frac{\partial L}{\partial z_i} = p_i - y_i $$

**CrossEntropyLoss (Multiclase, softmax + CE)**

$$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij}) $$

$$ P_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^{C} e^{z_{ik}}} $$

Derivada:

$$ \frac{\partial L}{\partial z_{ij}} = p_{ij} - y_{ij} $$

---

### Derivadas de funciones de activación (recordatorio)

- ReLU:

$$
f(x)=\max(0,x)
\qquad
f'(x)=
\begin{cases}
1 & \text{si } x > 0\\
0 & \text{si } x \le 0
\end{cases}
$$

- Sigmoid:

$$
f(x)=\frac{1}{1+\exp(-x)}
\qquad
f'(x)=f(x)\cdot(1-f(x))
$$

- Tanh:

$$
f(x)=\frac{\exp(2x)-1}{\exp(2x)+1}
\qquad
f'(x)=1-f(x)^2
$$

---


## Ejercicio 1 

$$
\hat{y} = \text{sigmoid}\big(\text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = BCELoss**

$$
x=
\begin{bmatrix}
0.5636 & 0.7666 \\
0.1621 & 0.4935 \\
\end{bmatrix}
\qquad
y \;(\text{binario})=
\begin{bmatrix}
1 \\
0 \\
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

**loss = BCELoss**

$$
x=
\begin{bmatrix}
-0.3536 & -0.3121 \\
0.8437 & 0.1697 \\
0.8841 & 0.3792 \\
\end{bmatrix}
\qquad
y \;(\text{binario})=
\begin{bmatrix}
0 \\
0 \\
1 \\
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

**loss = BCELoss**

$$
x=
\begin{bmatrix}
0.5636 & 0.7666 \\
0.1621 & 0.4935 \\
\end{bmatrix}
\qquad
y \;(\text{binario})=
\begin{bmatrix}
0 \\
0 \\
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

**loss = BCELoss**

$$
x=
\begin{bmatrix}
-0.3536 & -0.3121 & 0.8437 \\
0.1697 & 0.8841 & 0.3792 \\
\end{bmatrix}
\qquad
y \;(\text{binario})=
\begin{bmatrix}
1 \\
1 \\
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

**loss = BCELoss**

$$
x=
\begin{bmatrix}
0.2 & 0.5 & 0.8 \\
0.7 & 0.1 & 0.4 \\
\end{bmatrix}
\qquad
y \;(\text{binario})=
\begin{bmatrix}
0 \\
1 \\
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

**loss = CrossEntropyLoss **

$$
x=
\begin{bmatrix}
0.3 & 0.7 \\
0.8 & 0.2 \\
0.5 & 0.9 \\
\end{bmatrix}
\qquad
y \;(\text{one-hot})=
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
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
0.5 & 0.3 & -0.2 \\
-0.1 & 0.4 & 0.6 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.0 & 0.05 & -0.05 \\
\end{bmatrix}
$$


---

## Ejercicio 7 

$$
\hat{y} = \text{sigmoid}\big(\text{relu}( \text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}) \cdot W^{(3)} + b^{(3)}\big)
$$

**lr = 0.01**

**loss = CrossEntropyLoss i**


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
y \;(\text{one-hot})=
\begin{bmatrix}
0 & 0 & 1 \\
0 & 0 & 1 \\
0 & 0 & 1 \\
0 & 0 & 1 \\
0 & 0 & 1 \\
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
0.4 & 0.2 & -0.1 \\
-0.3 & 0.5 & 0.7 \\
\end{bmatrix}
\qquad
b^{(3)}=
\begin{bmatrix}
0.1 & -0.1 & 0.0 \\
\end{bmatrix}
$$


---

## Ejercicio 8 

$$
\hat{y} = \text{tanh}\big(\text{tanh}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.02**

**loss = CrossEntropyLoss **

Etiquetas convertidas:

$$
x=
\begin{bmatrix}
0.45 & -0.12 \\
-0.33 & 0.88 \\
0.77 & 0.05 \\
\end{bmatrix}
\qquad
y \;(\text{one-hot})=
\begin{bmatrix}
0 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
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
-0.4 & 0.2 & 0.5 \\
0.3 & -0.1 & -0.2 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
-0.05 & 0.05 & 0.0 \\
\end{bmatrix}
$$

---

## Ejercicio 9 

$$
\hat{y} = \text{relu}\big(\text{sigmoid}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.005**

**loss = CrossEntropyLoss **

$$
x=
\begin{bmatrix}
0.12 & 0.34 & 0.56 \\
0.78 & 0.90 & 0.11 \\
0.22 & 0.44 & 0.66 \\
\end{bmatrix}
\qquad
y \;(\text{one-hot})=
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 0 & 1 \\
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
0.25 & 0.15 & -0.05 \\
-0.35 & 0.45 & 0.55 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.0 & 0.05 & -0.05 \\
\end{bmatrix}
$$

---

## Ejercicio 10 

$$
\hat{y} = \text{sigmoid}\big(\text{relu}(x \cdot W^{(1)} + b^{(1)}) \cdot W^{(2)} + b^{(2)}\big)
$$

**lr = 0.01**

**loss = CrossEntropyLoss **


$$
x=
\begin{bmatrix}
0.33 & 0.66 \\
0.11 & 0.22 \\
0.77 & 0.88 \\
0.44 & 0.55 \\
\end{bmatrix}
\qquad
y \;(\text{one-hot})=
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
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
0.45 & 0.25 & -0.15 \\
-0.35 & 0.55 & 0.65 \\
\end{bmatrix}
\qquad
b^{(2)}=
\begin{bmatrix}
0.02 & -0.02 & 0.0 \\
\end{bmatrix}
$$

