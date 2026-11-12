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
# Computer Vision

![w:900](./imagenes/cv.png)

---
# La Carrera por la Visión Artificial

![bg  left:40% width:90% ](imagenes/carrera.png)

- Kunihiko Fukushima (1980): Neocognitron
- Yann LeCun (1998): LeNet-5
- Alex Krizhevsky (2012): AlexNet
- Kaiming He (2015): ResNet

---

# El Hola Mundo del Deep Learning

![bg  left:40% width:80% ](imagenes/mnist.png)


---

# Tensores

![bg  left:40% width:80% ](imagenes/tensores.png)

- Rangos 
- Broadcasting 
- reshape

--- 

# Problemas de clasificación binaria:

![w:900](imagenes/cb.png)


---

# BCELoss

![bg  left:40% width:80% ](imagenes/bce.png)

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left(
y_i \log(p_i)
+
(1-y_i)\log(1-p_i)
\right)
$$

$$ p_i = \frac{1}{1 + e^{-z_i}} $$

Derivada:

$$ \frac{\partial L}{\partial z_i} = \frac{p_i - y_i}{N} $$


---

# Accuracy


![bg  left:40% width:80% ](imagenes/acc.png)

$$ Accuracy = \frac{TP + TN}{Total} $$

---
# Inicialización de pesos

![bg left:40% width:90%](imagenes/inicializacion.png)

$$
y = \sigma(x \cdot W + b)
$$

- Si $W$ es muy grande, las activaciones también serán muy grandes.
- Si $W$ es muy pequeño, las activaciones también serán muy pequeñas.

> [torch init](https://docs.pytorch.org/docs/main/nn.init.html)

---

# 1) Random Initialization

$$
W_{ij}\sim\mathcal{N}(0,\varepsilon^2),\qquad\varepsilon>0
$$

![bg left:40% width:90%](imagenes/randominit.png)

- Es un método clásico que actualmente está en desuso, ya que no controla la varianza de las activaciones en redes profundas.

---

# 2) Zero Initialization

![bg left:40% width:90%](imagenes/zeroinit.png)

$$
b = 0
$$

- Se utiliza para inicializar los **bias**, pero no los pesos, salvo en algunos casos muy particulares.

---

# 3) Xavier/Glorot Initialization

![bg left:40% width:90%](imagenes/xavierinit.png)

$$
W \sim \mathcal{N}\left(0,\frac{2}{n_{in}+n_{out}}\right)
$$

$$
W \sim \mathcal{U}\left(
-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}
\right)
$$

- Mantiene aproximadamente la varianza entre capas.
- Recomendada para **tanh** y **sigmoid**.

---

# 4) He / Kaiming Initialization

![bg left:40% width:90%](imagenes/kaimingheinit.png)

$$
W \sim \mathcal{N}\left(0,\frac{2}{(1+a^2)n_{in}}\right),\quad a=0\ \text{en ReLU}
$$

$$
W \sim \mathcal{U}\left(
-\sqrt{\frac{6}{(1+a^2)n_{in}}},
\sqrt{\frac{6}{(1+a^2)n_{in}}}
\right)
$$

- Conserva aproximadamente la varianza con **ReLU** y sus variantes.
- **Leaky ReLU / PReLU**: $a$ es la pendiente negativa.

---
# Normalización en Redes Neuronales

![w:900](./imagenes/dlnorm.png)

---

# 1) Batch Normalization

![bg left:40% width:90%](imagenes/batchnorm.png)

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}
\qquad
y_i=\gamma\hat{x}_i+\beta
$$

- Media y varianza del **mini-batch**.
- Muy utilizada en CNN.
- Menos eficaz con batches pequeños.

---
### Derivada de Batch Normalization

Sea $\delta_i=\frac{\partial L}{\partial y_i}$.

La derivada respecto a la entrada es

$$
\frac{\partial L}{\partial x_i}
= \frac{\gamma}{m\sqrt{\sigma_B^2+\varepsilon}}
\left[
m\delta_i - \sum_j\delta_j - \hat{x}_i
\sum_j
\delta_j\hat{x}_j \right]
$$

Además,

$$
\frac{\partial L}{\partial\gamma}
= \sum_i\delta_i\hat{x}_i
\qquad
\frac{\partial L}{\partial\beta}
=\sum_i\delta_i
$$

> Observe que el gradiente de una muestra depende de todas las muestras del mini-batch.


---

# 2) Layer Normalization

![bg left:40% width:90%](imagenes/layernorm.png)

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
\qquad
y=\gamma\odot\hat{x}+\beta
$$

- Media y varianza por **muestra**.
- No depende del tamaño del batch.
- Estándar en Transformers.

---

### Derivada de Layer Normalization

$$
\frac{\partial L}{\partial x_i} = \frac{1}{d\sqrt{\sigma^2+\varepsilon}}
\left[
d\gamma_i\delta_i - \sum_j\gamma_j\delta_j - \hat{x}_i
\sum_j
\gamma_j\delta_j\hat{x}_j \right]
$$

$$
\frac{\partial L}{\partial\gamma_i} =
\sum_n\delta_{n,i}\hat{x}_{n,i}
$$

$$
\frac{\partial L}{\partial\beta_i} = \sum_n\delta_{n,i}
$$

> Entrada: características de una muestra. Parámetros: batch.

---

# 3) Group Normalization

![bg left:40% width:90%](imagenes/groupnorm.png)

$$
\mu_g=\frac{1}{m_g}\sum_{i\in g}x_i
$$

$$
\sigma_g^2=\frac{1}{m_g}\sum_{i\in g}(x_i-\mu_g)^2
$$

$$
\hat{x}=\frac{x-\mu_g}{\sqrt{\sigma_g^2+\varepsilon}}
\qquad
y=\gamma\odot\hat{x}+\beta
$$

- $G$ grupos de canales, $m_g=\frac{C}{G}HW$.
- Normaliza cada grupo por separado.
- Funciona con batches pequeños.

---

### Derivada de Group Normalization

$$
\frac{\partial L}{\partial x_i} = \frac{1}{m_g\sqrt{\sigma_g^2+\varepsilon}}
\left[
m_g\gamma_i\delta_i -
\sum_{j\in g}\gamma_j\delta_j - \hat{x}_i
\sum_{j\in g}
\gamma_j\delta_j\hat{x}_j
\right]
$$

$$
\frac{\partial L}{\partial\gamma_c} = \sum_{n,h,w}\delta_{n,c,h,w}\hat{x}_{n,c,h,w}
$$

$$
\frac{\partial L}{\partial\beta_c} =
\sum_{n,h,w}\delta_{n,c,h,w}
$$

> Entrada: elementos del grupo. Parámetros: batch y posiciones espaciales, por canal.

---

# 4) RMS Normalization

![bg left:40% width:90%](imagenes/rmsnorm.png)

$$
\text{RMS}(x)
= \sqrt{\frac1d\sum_{i=1}^{d}x_i^2+\varepsilon}
$$

$$
\hat{x} =
\frac{x}
{\text{RMS}(x)}
$$

$$
y=\gamma\odot\hat{x}
$$

- No calcula la media.
- Solo normaliza utilizando la magnitud RMS.
- Reduce el costo computacional.
- Muy utilizada en LLM modernos.

--- 

### Derivada de RMS Normalization

Sea

$$
R=\sqrt{\frac1d\sum_i x_i^2+\varepsilon}
\qquad
y=\gamma\odot\frac{x}{R}
$$

La derivada respecto a la entrada es

$$
\frac{\partial L}{\partial x} =
\frac{1}{R} \left(
I- \frac{xx^T}{dR^2} \right)
(\gamma\odot\delta)
$$

$$
\frac{\partial L}{\partial\gamma_i}
= \sum_n\delta_{n,i}\hat{x}_{n,i}
$$

> Backward más simple: no deriva la media ni la varianza.


---

# 5) Instance Normalization

![bg left:40% width:90%](imagenes/instancenorm.png)

$$
\mu_c=\frac1{HW}\sum_{h,w}x_{c,h,w}
$$

$$
\sigma_c^2=\frac1{HW}\sum_{h,w}(x_{c,h,w}-\mu_c)^2
$$

$$
\hat{x}=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}
\qquad
y=\gamma_c\hat{x}+\beta_c
$$

- Normaliza cada canal por imagen.
- Muy utilizada en Style Transfer y GANs.
- Equivale a GroupNorm cuando $G=C$.

---

### Derivada de Instance Normalization

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma_c}{HW\sqrt{\sigma_c^2+\varepsilon}}
\left[
HW\delta_i - \sum_j\delta_j -
\hat{x}_i
\sum_j
\delta_j\hat{x}_j
\right]
$$

$$
\frac{\partial L}{\partial\gamma_c}
= \sum_{n,h,w}\delta_{n,c,h,w}\hat{x}_{n,c,h,w}
$$

$$
\frac{\partial L}{\partial\beta_c} =
\sum_{n,h,w}\delta_{n,c,h,w}
$$

> Cada imagen y cada canal se derivan de forma independiente.

---
# Problemas de clasificación multiclase:

![w:900](imagenes/mcc.png)


---
# CrossEntropyLoss

![bg  left:40% width:80% ](imagenes/cel.png)


$$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij}) $$

$$ p_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^{C} e^{z_{ik}}} $$

Derivada:

$$ \frac{\partial L}{\partial z_{ij}} = \frac{p_{ij} - y_{ij}}{N} $$

--- 
# Softmax

![bg left:40% width:80%](imagenes/softmax.png)

$$
p_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
$$

Derivada:

$$
\frac{\partial p_j}{\partial z_i} =
\begin{cases}
p_j(1 - p_j) & \text{si } i = j \\
-p_j p_i & \text{si } i \neq j
\end{cases}
$$

---
