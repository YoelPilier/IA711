---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de PyTorch

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender que es un tensor y como se diferencia de un arreglo tradicional.
- Practicar operaciones basicas y movimiento entre CPU y GPU.
- Introducir gradientes, capas lineales y optimizacion.
- Reconocer las piezas de un ciclo de entrenamiento pequeno.

---

# Antes de empezar

Si PyTorch te confunde, casi siempre el problema esta en una de estas tres cosas: la forma del tensor, el tipo de dato o el dispositivo. Revisa eso antes de asumir que el modelo esta mal.

- Mira `shape` cuando una operacion falle.
- Mira `dtype` cuando el calculo no se comporte como esperas.
- Mira `device` cuando combines CPU y GPU.

---

# PyTorch y tensores

PyTorch usa tensores como estructura central. Son parecidos a los `ndarray` de NumPy, pero ademas pueden aprovechar GPU y calculo automatico de gradientes.

```python
import torch
from torch import nn
```

- Si ya entiendes NumPy, la forma de pensar los datos sera muy parecida.
- La gran diferencia aparece cuando quieres derivar y entrenar modelos.

---

# Crear tensores

```python
tensor_rango_0 = torch.tensor(1)
tensor_rango_1 = torch.tensor([1, 2, 3])
tensor_rango_2 = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(tensor_rango_0.shape)
print(tensor_rango_1.shape)
print(tensor_rango_2.shape)
```

- Revisa `shape`, `dtype` y `device`.
- El rango del tensor indica cuantas dimensiones tiene.

---

# Dispositivos

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
```

- `cpu` funciona siempre; `cuda` usa GPU si esta disponible.
- Modelo y datos deben estar en el mismo dispositivo para operar correctamente.

---

# Operaciones basicas

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

print(a + b)
print(a - b)
print(a * b)
print(a @ b.T)
```

- `*` hace producto elemento a elemento.
- `@` realiza producto matricial.
- En aprendizaje profundo es comun trabajar con `float32`.

---

# Gradientes con `autograd`

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x

y.backward()

print(x.grad)
```

- `requires_grad=True` activa el seguimiento de operaciones.
- `backward()` calcula derivadas automaticamente.
- Ese mecanismo es la base del entrenamiento por descenso de gradiente.

---

# No linealidad

```python
valores = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(torch.relu(valores))
print(torch.tanh(valores))
```

- Activaciones comunes: `ReLU`, `Tanh`, `Sigmoid`.
- Introducen no linealidad, sin la cual una red profunda se comportaria como una transformacion lineal.

---

# `nn.Linear` vs formula manual

```python
capa = nn.Linear(in_features=2, out_features=3)
x = torch.tensor([[1.0, 2.0]])

salida_1 = capa(x)
salida_2 = x @ capa.weight.T + capa.bias

print(torch.allclose(salida_1, salida_2))
```

- Una capa lineal aplica pesos y sesgo.
- El ejemplo muestra que `nn.Linear` no es magia: implementa una formula conocida.

---

# Forward, perdida y backward

```python
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[5.0], [10.0], [15.0]])

w = nn.Parameter(torch.randn(1, 1))
b = nn.Parameter(torch.zeros(1))

pred = x @ w + b
loss = ((pred - y) ** 2).mean()
loss.backward()

print(loss.item(), w.grad, b.grad)
```

- `pred` es la salida del modelo.
- `loss` mide que tan lejos esta la prediccion del objetivo.
- Luego `backward()` llena los gradientes de los parametros.

---

# Paso de optimizacion manual

```python
lr = 1e-3

with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

- Aqui se actualizan los parametros con los gradientes calculados.
- Luego se limpian los gradientes para la siguiente iteracion.

---

# Mini entrenamiento

```python
modelo = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

opt = torch.optim.SGD(modelo.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
```

- Un entrenamiento basico necesita modelo, funcion de perdida y optimizador.
- En cada epoca suele repetirse: forward, perdida, backward y `opt.step()`.

---

# Guardar y cargar modelos con `safetensors`

Cuando un modelo ya entreno o quieres seguir trabajando luego, necesitas guardar sus pesos. Una opcion comoda y segura es `safetensors`.

```python
from safetensors.torch import load_model, save_model
```

```python
modelo = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

save_model(modelo, "modelo.safetensors")
```

```python
modelo_cargado = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

load_model(modelo_cargado, "modelo.safetensors")
modelo_cargado.eval()
```

- `save_model(...)` guarda los pesos del modelo en un archivo `.safetensors`.
- `load_model(...)` carga esos pesos en otro modelo con la misma arquitectura.
- Si la arquitectura no coincide, la carga va a fallar o quedar incompleta.
- Despues de cargar, `eval()` es util si vas a usar el modelo para inferencia.

---

# Buenas practicas

- Verifica la forma de los tensores antes de entrenar.
- Mueve datos y modelo al mismo `device`.
- Usa `torch.no_grad()` cuando no necesites gradientes, por ejemplo en evaluacion.
- Si el modelo no aprende, revisa primero datos, escala y funcion de perdida.
- Guarda pesos con nombres claros para no confundir versiones de modelos.

---

# Ejercicios sugeridos

1. Crea tensores de rango 0, 1 y 2 e imprime sus formas.
2. Realiza suma, resta, producto elemento a elemento y producto matricial.
3. Calcula el gradiente de `y = x**2 + 2x` en varios valores de `x`.
4. Compara `torch.relu()` y `torch.tanh()` sobre el mismo tensor.
5. Reproduce el ejemplo `nn.Linear` y verifica el resultado manualmente.
6. Entrena un modelo lineal para aproximar `y = 3x + 2`.
7. Normaliza datos de entrada antes de entrenar y compara el comportamiento.
8. Crea una red pequena para convertir Celsius a Fahrenheit y observa como cambia la perdida por epoca.
9. Guarda el modelo entrenado con `save_model`, cargalo en una nueva instancia con `load_model` y compara las predicciones.

---
