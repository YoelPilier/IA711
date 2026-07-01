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
2. Calcular el error (*loss*).
3. Calcular las derivadas (*backward pass*).
4. Actualizar los pesos y el sesgo (*weight and bias update*).
5. Realizar una nueva predicción (*forward pass*) con los pesos y el sesgo actualizados.
6. Comparar el error antes y después de la actualización.
7. En clasificación, calcular la accuracy correspondiente.

---
# Información necesaria

## Convolución

$$
Y_{i,j}=\sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1}X_{i+m,j+n}K_{m,n}+b
$$

$$
H_{out} =
\left\lfloor
\frac{H_{in}+2P_h-D_h(K_h-1)-1}{S_h}
\right\rfloor
+1
$$

$$
W_{out} =
\left\lfloor
\frac{W_{in}+2P_w-D_w(K_w-1)-1}{S_w}
\right\rfloor
+1
$$

---

## Convolución Transpuesta

$$
O(iS_h+aD_h-P_h,\;jS_w+bD_w-P_w)
\mathrel{+}=
I(i,j)K(a,b)
$$

$$
a=0,\dots,K_h-1,
\quad
b=0,\dots,K_w-1
$$

$$
H_{out} =
(H_{in}-1)S_h -2P_h
+D_h(K_h-1)
+\mathrm{outPad}_h
+1
$$

---

## AutoEncoder

$$
z=f_\theta(x)
$$

$$
\hat{x}=g_\phi(z)
$$

$$
L=\|x-\hat{x}\|^2
$$

---

# Adaptive Average Pooling 2D

En estos ejercicios se usará:

$$
\text{AdaptiveAvgPool2d}(1,1)
$$

Esto significa sacar el promedio espacial de cada canal.

$$
a_c = \frac{1}{HW}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}X_{c,h,w}
$$

Si hay 2 canales, la salida queda:

$$
a=
\begin{bmatrix}
a_1 & a_2
\end{bmatrix}
$$

---
# Derivada Adaptive Average Pooling

Si:

$$
a_c = \frac{1}{HW}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}X_{c,h,w}
$$

Entonces:

$$
\frac{\partial L}{\partial X_{c,h,w}}=
\frac{1}{HW}
\frac{\partial L}{\partial a_c}
$$

---

# Ejercicio 1

$$
\hat{y}=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

**lr = 0.01**

**loss = MSE**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 1

$$
X=
\begin{bmatrix}
1.0 & 0.0 & 2.0 & 1.0\\
0.0 & 1.0 & 1.0 & 0.0\\
2.0 & 1.0 & 0.0 & 1.0\\
1.0 & 0.0 & 1.0 & 2.0
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
1.2
\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
0.2 & -0.1\\
0.4 & 0.3
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
-0.3 & 0.5\\
0.1 & -0.2
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
0.0 & 0.1
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.6\\
-0.4
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.05
\end{bmatrix}
$$

---
# Ejercicio 2

$$
\hat{y}=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

**lr = 0.005**

**loss = MAE**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 1

$$
X=
\begin{bmatrix}
0.5 & 1.0 & 0.0 & 1.5\\
1.0 & 0.0 & 0.5 & 0.0\\
0.0 & 1.5 & 1.0 & 0.5\\
1.0 & 0.5 & 0.0 & 1.0
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.7
\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
0.1 & 0.3\\
-0.2 & 0.4
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
0.5 & -0.4\\
0.2 & 0.1
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
-0.05 & 0.0
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.3\\
0.8
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
-0.1
\end{bmatrix}
$$

---
# Ejercicio 3

$$
\hat{y}=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

**lr = 0.02**

**loss = MSE**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 2

$$
X=
\begin{bmatrix}
1.0 & 2.0 & 0.0 & 1.0\\
0.5 & 1.0 & 1.5 & 0.0\\
2.0 & 0.0 & 1.0 & 0.5\\
0.0 & 1.0 & 0.5 & 2.0
\end{bmatrix}
\qquad
y=
\begin{bmatrix}
0.4 & 1.0
\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
0.2 & 0.0\\
0.1 & -0.3
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
-0.1 & 0.4\\
0.3 & 0.2
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
0.1 & -0.1
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.5 & -0.2\\
0.3 & 0.7
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.0 & 0.05
\end{bmatrix}
$$

---
# Ejercicio 4

$$
z=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

$$
p=\text{sigmoid}(z)
$$

**lr = 0.01**

**loss = BCELoss**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 1

$$
X=
\begin{bmatrix}
0.0 & 1.0 & 1.0 & 0.0\\
1.0 & 1.0 & 0.0 & 0.5\\
0.0 & 0.5 & 1.0 & 1.0\\
1.0 & 0.0 & 0.5 & 1.0
\end{bmatrix}
\qquad
y=\begin{bmatrix}1\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
0.4 & -0.2\\
0.1 & 0.3
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
-0.3 & 0.2\\
0.5 & -0.1
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
0.0 & 0.05
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.7\\
-0.6
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.1
\end{bmatrix}
$$

---
# Ejercicio 5

$$
z=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

$$
p=\text{softmax}(z)
$$

**lr = 0.005**

**loss = CrossEntropyLoss**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 3

$$
X=
\begin{bmatrix}
1.0 & 0.5 & 0.0 & 1.0\\
0.0 & 1.0 & 1.0 & 0.5\\
1.5 & 0.0 & 0.5 & 1.0\\
0.5 & 1.0 & 0.0 & 1.5
\end{bmatrix}
\qquad
y=\begin{bmatrix}0 & 1 & 0\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
0.3 & 0.1\\
-0.2 & 0.4
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
0.2 & -0.5\\
0.3 & 0.1
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
-0.1 & 0.0
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.2 & 0.5 & -0.3\\
-0.4 & 0.1 & 0.6
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.0 & 0.05 & -0.05
\end{bmatrix}
$$

---
# Ejercicio 6

$$
z=\text{Linear}\left(
\text{AdaptiveAvgPool2d}\left(
\text{ReLU}(X*K+b_c),(1,1)
\right)
\right)
$$

$$
p=\text{softmax}(z)
$$

**lr = 0.02**

**loss = CrossEntropyLoss**

**Conv2d**

- **in_channels** = 1
- **out_channels** = 2
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**AdaptiveAvgPool2d**

- **output_size** = (1,1)

**Linear**

- **in_features** = 2
- **out_features** = 3

$$
X=
\begin{bmatrix}
0.5 & 0.0 & 1.0 & 1.5\\
1.0 & 0.5 & 0.0 & 1.0\\
0.0 & 1.0 & 1.5 & 0.5\\
1.5 & 0.5 & 1.0 & 0.0
\end{bmatrix}
\qquad
y=\begin{bmatrix}0 & 0 & 1\end{bmatrix}
$$


$$
K^{(1)}=
\begin{bmatrix}
-0.2 & 0.4\\
0.3 & 0.1
\end{bmatrix}
\qquad
K^{(2)}=
\begin{bmatrix}
0.5 & -0.1\\
-0.3 & 0.2
\end{bmatrix}
$$

$$
b_c=
\begin{bmatrix}
0.05 & -0.05
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.6 & -0.2 & 0.1\\
0.3 & 0.4 & -0.5
\end{bmatrix}
\qquad
b=
\begin{bmatrix}
0.1 & -0.1 & 0.0
\end{bmatrix}
$$

---
# Ejercicio 7

$$
Z=\text{ConvTranspose2d}\left(
\text{ReLU}(\text{Conv2d}(X,K_e,b_e)),K_d,b_d
\right)
$$

$$
\hat{X}=\text{sigmoid}(Z)
$$

**lr = 0.01**

**loss = MSE**

**Conv2d encoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**ConvTranspose2d decoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **output_padding** = 0
- **dilation** = 1

$$
X=
\begin{bmatrix}
1.0 & 0.0 & 1.0\\
0.0 & 1.0 & 0.0\\
1.0 & 0.0 & 1.0
\end{bmatrix}
\qquad
Y=
\begin{bmatrix}
1.0 & 0.0 & 1.0\\
0.0 & 1.0 & 0.0\\
1.0 & 0.0 & 1.0
\end{bmatrix}
$$

$$
K_e=
\begin{bmatrix}
0.5 & -0.2\\
0.3 & 0.1
\end{bmatrix}
\qquad
b_e=\begin{bmatrix}0.0\end{bmatrix}
$$

$$
K_d=
\begin{bmatrix}
0.4 & 0.1\\
-0.3 & 0.2
\end{bmatrix}
\qquad
b_d=\begin{bmatrix}0.05\end{bmatrix}
$$

---
# Ejercicio 8

$$
Z=\text{ConvTranspose2d}\left(
\text{ReLU}(\text{Conv2d}(X,K_e,b_e)),K_d,b_d
\right)
$$

$$
\hat{X}=\text{sigmoid}(Z)
$$

**lr = 0.005**

**loss = MSE**

**Conv2d encoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**ConvTranspose2d decoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **output_padding** = 0
- **dilation** = 1

$$
X=
\begin{bmatrix}
1.0 & 0.2 & 0.9\\
0.1 & 0.8 & 0.2\\
0.9 & 0.1 & 1.0
\end{bmatrix}
\qquad
Y=
\begin{bmatrix}
1.0 & 0.0 & 1.0\\
0.0 & 1.0 & 0.0\\
1.0 & 0.0 & 1.0
\end{bmatrix}
$$


$$
K_e=
\begin{bmatrix}
0.2 & 0.4\\
-0.1 & 0.3
\end{bmatrix}
\qquad
b_e=\begin{bmatrix}-0.05\end{bmatrix}
$$

$$
K_d=
\begin{bmatrix}
0.3 & -0.2\\
0.5 & 0.1
\end{bmatrix}
\qquad
b_d=\begin{bmatrix}0.0\end{bmatrix}
$$

---
# Ejercicio 9

$$
Z=\text{ConvTranspose2d}\left(
\text{ReLU}(\text{Conv2d}(X,K_e,b_e)),K_d,b_d
\right)
$$

$$
\hat{X}=\text{sigmoid}(Z)
$$

**lr = 0.01**

**loss = BCELoss**

**Conv2d encoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**ConvTranspose2d decoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 2x2
- **stride** = 1
- **padding** = 0
- **output_padding** = 0
- **dilation** = 1

$$
X=
\begin{bmatrix}
0.0 & 1.0 & 0.0\\
1.0 & 1.0 & 1.0\\
0.0 & 1.0 & 0.0
\end{bmatrix}
\qquad
Y=
\begin{bmatrix}
0 & 1 & 0\\
1 & 1 & 1\\
0 & 1 & 0
\end{bmatrix}
$$


$$
K_e=
\begin{bmatrix}
0.3 & -0.1\\
0.4 & 0.2
\end{bmatrix}
\qquad
b_e=\begin{bmatrix}0.1\end{bmatrix}
$$

$$
K_d=
\begin{bmatrix}
0.2 & 0.5\\
-0.4 & 0.3
\end{bmatrix}
\qquad
b_d=\begin{bmatrix}-0.1\end{bmatrix}
$$

---
# Ejercicio 10

$$
Z=\text{ConvTranspose2d}\left(
\text{ReLU}(\text{Conv2d}(X,K_e,b_e)),K_d,b_d
\right)
$$

$$
\hat{X}=\text{sigmoid}(Z)
$$

**lr = 0.02**

**loss = MAE**

**Conv2d encoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 3x3
- **stride** = 1
- **padding** = 0
- **dilation** = 1

**ConvTranspose2d decoder**

- **in_channels** = 1
- **out_channels** = 1
- **kernel_size** = 3x3
- **stride** = 1
- **padding** = 0
- **output_padding** = 0
- **dilation** = 1

$$
X=
\begin{bmatrix}
0.0 & 1.0 & 0.5 & 1.0\\
1.0 & 0.0 & 1.0 & 0.5\\
0.5 & 1.0 & 0.0 & 1.0\\
1.0 & 0.5 & 1.0 & 0.0
\end{bmatrix}
$$

$$
Y=
\begin{bmatrix}
0.0 & 1.0 & 0.5 & 1.0\\
1.0 & 0.0 & 1.0 & 0.5\\
0.5 & 1.0 & 0.0 & 1.0\\
1.0 & 0.5 & 1.0 & 0.0
\end{bmatrix}
$$


$$
K_e=
\begin{bmatrix}
0.2 & -0.1 & 0.3\\
0.4 & 0.0 & -0.2\\
0.1 & 0.5 & 0.2
\end{bmatrix}
\qquad
b_e=\begin{bmatrix}0.0\end{bmatrix}
$$

$$
K_d=
\begin{bmatrix}
0.3 & 0.1 & -0.2\\
0.0 & 0.4 & 0.2\\
-0.1 & 0.3 & 0.5
\end{bmatrix}
\qquad
b_d=\begin{bmatrix}0.05\end{bmatrix}
$$

---
