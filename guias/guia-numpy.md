---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de NumPy

[Documentación oficial de NumPy](https://numpy.org/doc/)

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender que es un `ndarray` y por que NumPy es tan usado en ciencia de datos.
- Crear arreglos, revisar sus dimensiones y operar con ellos.
- Practicar broadcasting, indexacion, matrices y reducciones.
- Relacionar arreglos numericos con datos reales como imagenes.

---

# Antes de empezar

Si solo necesitas recordar una operacion o entender por que un arreglo no coincide con otro, entra directo a la seccion correspondiente y prueba los ejemplos con tus propios datos.

- Mira `shape` antes de suponer.
- Mira `dtype` antes de mezclar tipos.
- Prefiere operaciones vectorizadas antes de escribir bucles.

---

# Que es NumPy

NumPy es la base numerica de gran parte del ecosistema cientifico de Python. Su estructura principal es `ndarray`, un arreglo multidimensional homogeneo y eficiente.

```python
import numpy as np
```

- NumPy guarda datos del mismo tipo de forma compacta.
- Esa organizacion lo vuelve mucho mas rapido que usar listas para calculo numerico intensivo.

---

# Crear arreglos

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.zeros((2, 3))
d = np.ones((2, 2))
```

- Revisa siempre `shape`, `ndim` y `dtype`.

```python
print(a.shape, a.ndim, a.dtype)
print(b.shape, b.ndim, b.dtype)
```

- `shape` describe la forma del arreglo.
- `ndim` indica cuantas dimensiones tiene.
- `dtype` dice como se almacenan los datos.

---

# Operaciones con escalares y vectores

```python
a = np.array([1, 2, 3])
b = np.array([4, 3, 6])

print(a + b)
print(a - b)
print(a * b)   # producto elemento a elemento
print(a @ b)   # producto punto
print(a / b)
```

- La mayoria de operaciones se aplican elemento a elemento.
- `@` representa producto punto o matricial segun la forma de los arreglos.

---

# Broadcasting

Broadcasting es la regla que permite combinar arreglos con formas compatibles sin copiar manualmente los datos.

```python
vector = np.array([1, 2, 3])
escalar = np.array(4)

print(vector + escalar)
print(vector * escalar)
print(vector ** escalar)
```

- Un escalar puede actuar sobre todos los elementos del vector.
- Tambien se puede expandir una fila o columna si las dimensiones son compatibles.

---

# Reducciones y estadisticas

Muchas tareas de preparacion de datos consisten en resumir informacion: sumar, promediar, buscar maximos o medir dispersion.

```python
datos = np.array([2, 4, 6, 8, 10])

print(np.sum(datos))
print(np.mean(datos))
print(np.std(datos))
print(np.min(datos), np.max(datos))
```

- Estas funciones son muy comunes antes de entrenar modelos o comparar conjuntos de datos.

---

# Norma, normalizacion y z-score

```python
a = np.array([1, 2, 3], dtype=float)

norma = np.linalg.norm(a)
normalizado = a / norma
zscore = (a - np.mean(a)) / np.std(a)

print(norma)
print(normalizado)
print(zscore)
```

- La norma mide magnitud.
- Normalizar suele dejar un vector con longitud 1.
- El z-score centra los datos y los reescala segun su desviacion estandar.

---

# Matrices

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

print(a + b)
print(a * b)   # Hadamard
print(a @ b)   # producto matricial
print(a.T)     # transpuesta
```

- El producto de Hadamard multiplica posicion con posicion.
- El producto matricial mezcla filas con columnas y aparece mucho en redes neuronales.
- `.T` intercambia filas por columnas.

---

# Formas e indexacion

```python
x = np.arange(12)
matriz = x.reshape(3, 4)

print(matriz[0])
print(matriz[:, 1])
print(matriz[1:, 2:])
```

- `reshape()` reorganiza datos sin cambiar su contenido.
- `:` significa "toma todo en ese eje".
- Saber hacer slicing ahorra muchos bucles manuales.

---

# NumPy y datos de imagen

```python
from datasets import load_dataset

dataset = load_dataset("fashion_mnist")
imagen = np.array(dataset["train"][0]["image"])

print(imagen.shape)
```

- Una imagen en escala de grises puede verse como una matriz.
- Ese paso conecta NumPy con vision por computadora y preprocesamiento.

---

# Buenas practicas

- Usa operaciones vectorizadas antes de escribir bucles manuales.
- Revisa `shape` cada vez que algo no salga como esperas.
- Si un calculo requiere decimales, usa `dtype=float`.
- Diferencia entre `*` y `@`: una confusion aqui cambia por completo el resultado.

---

# Ejercicios sugeridos

1. Crea un vector de 10 elementos y calcula suma, media y desviacion estandar.
2. Genera dos vectores y calcula su producto punto.
3. Aplica broadcasting para sumar un escalar a todos los elementos de una matriz.
4. Normaliza un vector y verifica que su norma sea cercana a 1.
5. Aplica z-score a una lista de notas y compara los valores originales con los transformados.
6. Crea dos matrices `3x3` y calcula suma, Hadamard, producto matricial y transpuesta.
7. Usa `reshape()` para convertir un vector de 16 elementos en una matriz `4x4` y extrae una submatriz central.
8. Carga una imagen y convierte sus pixeles en un arreglo con NumPy; luego imprime su forma y tipo de dato.

---
