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

---
