---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de Pandas

[Documentación oficial de Pandas](https://pandas.pydata.org/docs/)

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender las estructuras `Series` y `DataFrame`.
- Crear, inspeccionar y seleccionar datos tabulares.
- Limpiar nulos y duplicados antes de analizar o modelar.
- Agrupar, transformar y exportar resultados.
- Filtrar y transformar datasets de Hugging Face y convertirlos a Pandas.

---

# Antes de empezar

Cuando trabajes con una tabla y no sepas por donde empezar, primero inspecciona. Casi siempre conviene mirar unas cuantas filas, revisar tipos y confirmar si hay valores faltantes antes de transformar nada.

- Empieza por `head()`, `info()` y `describe()`.
- Despues filtra o limpia con mas contexto.
- Si algo cambia mucho el dataset, vuelve a inspeccionarlo.

---

# Que es Pandas

Pandas se usa para manipular datos tabulares. La documentacion oficial destaca dos estructuras clave:

- `Series`: una columna o secuencia etiquetada.
- `DataFrame`: una tabla bidimensional con filas y columnas.

```python
import pandas as pd
```

- Pandas es ideal cuando los datos vienen de CSV, Excel, APIs o formularios.
- Su fortaleza no es solo guardar datos, sino permitir filtrarlos y resumirlos rapido.

---

# Crear un `DataFrame`

```python
df = pd.DataFrame({
    "Nombre": ["Naruto", "One Piece", "Death Note"],
    "Demografia": ["Shonen", "Shonen", "Seinen"],
    "Valoracion": [8.5, 9.0, 8.8]
})

print(df)
```

- Un diccionario de listas es una forma comun de construir tablas.
- Cada clave se convierte en una columna.

---

# Inspeccion inicial

Antes de transformar datos conviene mirar su tamano, tipos y posibles problemas.

```python
df.head()
df.tail()
df.info()
df.describe()
```

- `head()` y `tail()` muestran filas del inicio y del final.
- `info()` resume tipos y valores no nulos.
- `describe()` da estadisticas rapidas sobre columnas numericas.

---

# Seleccion de datos

```python
df["Nombre"]          # columna
df.iloc[0]            # fila por posicion
df.loc[0, "Nombre"]   # celda por etiqueta
df.iloc[:2, :2]       # subconjunto
```

- `loc` trabaja por etiquetas.
- `iloc` trabaja por posiciones.
- Saber seleccionar bien hace que filtros y transformaciones sean mucho mas faciles de leer.

---

# Crear y transformar columnas

```python
df["Popularidad"] = ["Alta", "Alta", "Media"]
df["Anio"] = [2002, 1999, 2006]

df["Popularidad"] = df["Valoracion"].apply(
    lambda v: "Muy alta" if v > 8.9 else "Alta"
)
```

- Puedes crear columnas desde listas, operaciones matematicas o reglas.
- `apply()` es util para reglas simples fila por fila o valor por valor.

---

# Ordenar y filtrar

```python
df.sort_values(by="Valoracion", ascending=False)

df[df["Demografia"] == "Shonen"]

df.query('Demografia == "Shonen" and Valoracion >= 8.5')
```

- Filtrar permite quedarte solo con el subconjunto relevante.
- `query()` vuelve los filtros complejos mas legibles.

---

# Tipos, nulos y duplicados

```python
df["Demografia"] = df["Demografia"].astype("string")

df.isnull().sum()
df.fillna(0, inplace=True)

df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

- Revisar nulos y duplicados es una parte basica de limpieza.
- `astype(...)` ayuda a fijar tipos consistentes.
- Antes de usar `fillna()` conviene pensar si `0` realmente tiene sentido en ese contexto.

---

# Agrupacion y codificacion

```python
resumen = df.groupby("Demografia")["Valoracion"].describe()

dummies = pd.get_dummies(df["Demografia"], drop_first=True)
df = pd.concat([df, dummies], axis=1)
```

- `groupby()` resume grupos y permite responder preguntas agregadas.
- `get_dummies()` convierte categorias en columnas numericas.
- Estas operaciones aparecen mucho en preparacion de datos para modelos.

---

# Muestreo, archivos y graficos

```python
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

df.to_csv("animes.csv", index=False)
df = pd.read_csv("animes.csv")

df.plot.bar(x="Nombre", y="Valoracion", legend=False)
```

- `sample()` ayuda a separar conjuntos pequenos de entrenamiento y prueba.
- CSV es uno de los formatos mas comunes al trabajar con Pandas.
- Un `DataFrame` puede producir graficos basicos rapidamente.

---

# Hugging Face Datasets: cargar e inspeccionar

```python
from datasets import load_dataset

dataset = load_dataset("ylecun/mnist")
train_data = dataset["train"]
test_data = dataset["test"]

print(train_data.column_names)  # ["image", "label"]
print(len(train_data))
print(train_data[0]["label"])
```

- `dataset` contiene las particiones; `train_data` es un `Dataset`.
- Datasets tiene su propia API: no es un `DataFrame` de Pandas.
- Las operaciones siguientes devuelven un nuevo dataset; guarda el resultado.

---

# `filter`: filtrar ejemplo por ejemplo

```python
# Conservar solo los digitos 0 y 1 para clasificacion binaria.
binario = train_data.filter(
    lambda ejemplo: ejemplo["label"] in [0, 1]
)

print(len(train_data), len(binario))
print(binario[0]["label"])
```

- La funcion recibe un ejemplo como diccionario.
- Debe devolver `True` para conservarlo o `False` para descartarlo.
- La misma regla puede aplicarse a `test_data` sin modificar el original.

---

# `filter` con `batched=True`

```python
binario_batch = train_data.filter(
    lambda batch: [label in [0, 1] for label in batch["label"]],
    batched=True,
    batch_size=1000,
)

print(len(binario_batch))  # Misma cantidad que en binario.
```

- La funcion recibe un diccionario de listas, no un solo ejemplo.
- Debe devolver un booleano por cada ejemplo del lote.
- `batch_size` controla cuantos ejemplos recibe cada llamada; no es el batch de entrenamiento.
- `batched=True` no activa multiprocesamiento; `num_proc` es una opcion distinta.

---

# `map`: crear o transformar columnas

```python
con_nombre = binario.map(
    lambda ejemplo: {"nombre": "cero" if ejemplo["label"] == 0 else "uno"}
)

print(con_nombre.column_names)  # ["image", "label", "nombre"]
print(con_nombre[0]["nombre"])
```

- `filter` decide que filas conservar; `map` transforma su contenido.
- La funcion devuelve un diccionario con las columnas nuevas o actualizadas.
- Las columnas originales se conservan si no pides eliminarlas.

---

# `map` con `batched=True`

```python
con_nombre_batch = binario.map(
    lambda batch: {
        "nombre": ["cero" if label == 0 else "uno" for label in batch["label"]]
    },
    batched=True,
    batch_size=1000,
)
```

- En este ejemplo, cada lista devuelta tiene un valor por ejemplo de entrada.
- Un `map` por lotes devuelve columnas; un `filter` por lotes devuelve booleanos.
- Es util para preparar varias muestras en una misma llamada.

---

# Seleccionar, mezclar y ordenar

```python
# Un subconjunto pequeno para probar codigo.
primeros = train_data.select(range(min(100, len(train_data))))

# Mezclar antes de seleccionar para obtener una muestra aleatoria.
muestra = train_data.shuffle(seed=42).select(
    range(min(100, len(train_data)))
)

ordenados = binario.sort("label")
```

- `select` recibe posiciones y devuelve otro `Dataset`.
- `train_data[:5]` devuelve un diccionario con listas de valores.
- `seed=42` permite repetir la misma mezcla.

---

# Separar entrenamiento y validacion

```python
particiones = binario.train_test_split(
    test_size=0.2, seed=42, stratify_by_column="label"
)
train_binario = particiones["train"]
val_binario = particiones["test"]

print(len(train_binario), len(val_binario))
```

- Aqui la clave `"test"` se usa como validacion del conjunto de entrenamiento.
- `stratify_by_column` mantiene aproximadamente la proporcion de clases.
- MNIST define `label` como `ClassLabel`, el tipo que requiere esta estratificacion.
- El `test_data` original queda reservado para la evaluacion final.

---

# Columnas y conversion a Pandas

```python
from datasets import Dataset

# Para analizar etiquetas no necesitamos la columna de imagenes.
tabla = con_nombre.select_columns(["label", "nombre"])
tabla = tabla.rename_column("label", "digito")
df_mnist = tabla.to_pandas()

print(df_mnist["digito"].value_counts())
de_vuelta = Dataset.from_pandas(df_mnist, preserve_index=False)
```

- `remove_columns("image")` es otra forma de excluir las imagenes.
- `to_pandas()` materializa la tabla en memoria: convierte solo lo necesario.
- `preserve_index=False` evita agregar el indice como columna al volver a Datasets.

---

# Referencias: Hugging Face Datasets

- [Procesamiento: filter, map, seleccion y particiones](https://huggingface.co/docs/datasets/process).
- [Conversion e integracion con Pandas](https://huggingface.co/docs/datasets/use_with_pandas).

---

# Buenas practicas

- Inspecciona los datos antes de limpiarlos o modelarlos.
- Evita encadenar demasiadas transformaciones si luego sera dificil depurar.
- Usa nombres de columnas claros y consistentes.
- Revisa siempre cuantos registros quedan despues de filtrar o eliminar filas.

---

# Ejercicios sugeridos

1. Crea un `DataFrame` con al menos 8 filas sobre peliculas, libros o videojuegos.
2. Aplica `head()`, `tail()`, `info()` y `describe()` sobre ese conjunto.
3. Filtra solo los registros con nota mayor a 8 y ordenalos de mayor a menor.
4. Agrega una columna categorica creada con `apply()`.
5. Inserta valores nulos y practica `isnull()`, `dropna()` y `fillna()`.
6. Duplica algunas filas y luego elimina duplicados.
7. Agrupa por una variable categorica y resume la nota promedio.
8. Exporta el resultado final a CSV y genera un grafico de barras con una de sus columnas numericas.
9. Filtra MNIST para conservar los digitos 0 y 1, con y sin `batched=True`. Compara las etiquetas obtenidas.
10. Agrega una columna con `map`, separa entrenamiento y validacion y cuenta las clases con Pandas.

---
