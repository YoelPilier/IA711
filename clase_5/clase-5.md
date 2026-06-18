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

$$ P_i = \frac{1}{1 + e^{-z_i}} $$

Derivada:

$$ \frac{\partial L}{\partial z_i} = p_i - y_i $$


---

# Accuracy


![bg  left:40% width:80% ](imagenes/acc.png)

$$ Accuracy = \frac{TP + TN}{Total} $$

---


# Problemas de clasificación multiclase:

![w:900](imagenes/mcc.png)


---
# CrossEntropyLoss

![bg  left:40% width:80% ](imagenes/cel.png)


$$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij}) $$

$$ P_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^{C} e^{z_{ik}}} $$

Derivada:

$$ \frac{\partial L}{\partial z_{ij}} = p_{ij} - y_{ij} $$

--- 

# TBU 
