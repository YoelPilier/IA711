---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de Matplotlib

[Documentación oficial de Matplotlib](https://matplotlib.org/stable/contents.html)

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender la relacion entre `Figure` y `Axes`.
- Crear graficos basicos para explorar y comunicar datos.
- Personalizar titulos, etiquetas, colores y distribucion.
- Mostrar un batch de imagenes de MNIST con sus etiquetas.
- Guardar figuras para reportes o presentaciones.

---

# Antes de empezar

Cuando dudes entre varios graficos, no pienses primero en el color o en el estilo: piensa en la pregunta que quieres responder. El grafico correcto suele salir de ahi.

- Si comparas categorias, prueba barras.
- Si quieres ver relacion entre dos variables, prueba dispersion.
- Si quieres ver distribucion, prueba histograma o boxplot.

---

# Que es Matplotlib

Matplotlib es una libreria para visualizacion de datos en Python. La documentacion oficial organiza el trabajo alrededor de una `Figure` y uno o mas `Axes`.

```python
import matplotlib.pyplot as plt
```

- `Figure`: contenedor general de la visualizacion.
- `Axes`: zona concreta donde se dibuja un grafico.

---

# Primer grafico

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

- `plot(...)` crea una grafica de linea.
- `plt.show()` muestra la figura al final del script o del notebook.

---

# Etiquetas y personalizacion

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 8], label="Crecimiento")
ax.set_title("Grafico de linea")
ax.set_xlabel("Epoca")
ax.set_ylabel("Valor")
ax.grid(True)
ax.legend()
plt.show()
```

- Un buen grafico debe decir que se esta viendo sin necesidad de explicacion externa.
- Titulo, ejes y leyenda vuelven la figura interpretable.

---

# Tipos de graficos utiles

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].bar(["A", "B", "C"], [5, 7, 4])
axs[0, 1].scatter([1, 2, 3], [2, 5, 3])
axs[1, 0].hist([1, 1, 2, 3, 3, 3, 4])
axs[1, 1].pie([40, 35, 25], labels=["Train", "Val", "Test"])

plt.tight_layout()
plt.show()
```

- `bar`: compara categorias.
- `scatter`: muestra relacion entre dos variables.
- `hist`: ayuda a ver distribuciones.
- `pie`: conviene usarlo solo cuando hay pocas categorias claras.

---

# Ejemplos alineados con el laboratorio

```python
import pandas as pd

datos = pd.DataFrame({
    "Anime": ["Naruto", "One Piece", "Death Note"],
    "Valoracion": [8.5, 9.0, 8.8]
})

datos.plot.bar(x="Anime", y="Valoracion", legend=False)
plt.xlabel("Anime")
plt.ylabel("Valoracion")
plt.title("Valoracion por anime")
plt.show()
```

- Muchos analisis en clase parten de un `DataFrame` y terminan en un grafico.
- Aunque uses `pandas.plot(...)`, debajo sigue trabajando Matplotlib.

---

# Dispersion, histogramas y cajas

```python
df = pd.DataFrame({
    "Animeid": [1, 2, 3, 4],
    "Valoracion": [8.5, 9.0, 8.8, 9.3]
})

df.plot.scatter(x="Animeid", y="Valoracion")
df["Valoracion"].plot.hist()
df[["Valoracion"]].plot.box()
plt.show()
```

- `scatter`: relacion entre variables.
- `hist`: distribucion de frecuencias.
- `box`: rango, mediana y posibles outliers.

---

# Imagenes con Matplotlib

```python
from datasets import load_dataset
from matplotlib import pyplot as plt

dataset = load_dataset("fashion_mnist")
plt.imshow(dataset["train"][0]["image"], cmap="gray")
plt.axis("off")
plt.show()
```

- `imshow()` permite visualizar matrices como imagenes.
- `cmap="gray"` es util para datos en escala de grises.

---

# MNIST: obtener un batch de imagenes

```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train_data = MNIST(
    root="./data", train=True, download=True, transform=ToTensor()
)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

images, labels = next(iter(train_loader))
print(images.shape)  # (16, 1, 28, 28): batch, canales, alto, ancho
print(labels.shape)  # (16,): una etiqueta por imagen
```

- `download=True` descarga MNIST si no esta en `./data`.
- `ToTensor()` convierte cada imagen a un tensor con valores entre 0 y 1.
- Un batch agrupa varias imagenes y sus etiquetas para procesarlas juntas.

---

# MNIST: mostrar el batch con Matplotlib

```python
cols = 4
rows = (len(images) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2), squeeze=False)

for i, ax in enumerate(axes.ravel()):
    ax.axis("off")
    if i < len(images):
        image = images[i].detach().cpu().squeeze(0)
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Etiqueta: {labels[i].item()}")

fig.tight_layout()
plt.show()
```

- `squeeze(0)` quita el canal unico: `(1, 28, 28)` pasa a `(28, 28)`.
- `axes.ravel()` permite recorrer la cuadricula; `squeeze=False` funciona tambien con una sola imagen.
- Los ejes sobrantes quedan ocultos si el batch no llena la cuadricula.

---

# Guardar figuras

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 2, 5])
fig.savefig("grafico.png", dpi=150, bbox_inches="tight")
```

- `savefig()` exporta la figura a archivo.
- `dpi` controla la resolucion.
- `bbox_inches="tight"` reduce margenes innecesarios.

---

# Buenas practicas

- Escoge el tipo de grafico segun la pregunta que quieres responder.
- Usa etiquetas legibles y evita sobrecargar con demasiados colores.
- Si comparas varias figuras, manten la misma escala cuando tenga sentido.
- `tight_layout()` ayuda a evitar que se monten los textos.

---

# Ejercicios sugeridos

1. Grafica una linea con la evolucion de una perdida por epoca.
2. Crea un grafico de barras con las calificaciones de cinco estudiantes.
3. Haz un histograma con 100 numeros aleatorios.
4. Genera un grafico de dispersion entre horas de estudio y nota final.
5. Replica con tus propios datos un `bar`, `scatter` y `boxplot`.
6. Carga una imagen en escala de grises y muestrala con `imshow`.
7. Crea una figura con dos subgraficos comparando train y test.
8. Exporta un grafico final a PNG con `savefig` y ajusta su resolucion.
9. Muestra un batch de MNIST con sus etiquetas. Prueba con `batch_size=1`, `10` y `16`.

---
