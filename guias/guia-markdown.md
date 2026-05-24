---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de Markdown

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender que es Markdown y por que aparece tanto en notebooks, repositorios y documentacion.
- Aprender la sintaxis basica para escribir texto con estructura.
- Usar encabezados, listas, codigo, enlaces, imagenes, citas y tablas.
- Introducir formulas con LaTeX para notas tecnicas y material de IA.

---

# Antes de empezar

Cuando necesites escribir una explicacion rapida en un notebook, una tarea en GitHub o una nota de proyecto, Markdown suele ser suficiente. La idea no es memorizar todo de golpe, sino reconocer los patrones mas usados y volver a ellos cuando haga falta.

- Empieza por encabezados, listas y codigo.
- Agrega tablas o formulas solo cuando realmente aporten claridad.
- Si dudas, escribe primero algo simple y luego mejora el formato.

---

# Que es Markdown

Markdown es un lenguaje de marcado ligero pensado para escribir texto facil de leer y facil de convertir a formatos como HTML. Se usa mucho en README, notebooks, foros, blogs y plataformas como GitHub, Kaggle o Colab.

```markdown
# Titulo

Este texto tiene **negrita**, *cursiva* y `codigo`.
```

- Su ventaja principal es que no obliga a usar editores complejos.
- Con pocas marcas puedes dar estructura al contenido.

---

# Encabezados

Los encabezados organizan el contenido por niveles. Mientras mas `#` uses, mas bajo sera el nivel del titulo.

```markdown
# Encabezado 1
## Encabezado 2
### Encabezado 3
#### Encabezado 4
##### Encabezado 5
```

# Encabezado 1
## Encabezado 2
### Encabezado 3
#### Encabezado 4
##### Encabezado 5

- Usa un orden logico: no saltes de `#` a `####` sin necesidad.
- Los encabezados ayudan a leer y tambien a navegar documentos largos.

---

# Enfasis de texto

Puedes resaltar palabras o frases para que una idea importante no se pierda dentro del parrafo.

```markdown
**Texto en negrita**
__Texto en negrita__
*Texto en cursiva*
_Texto en cursiva_
```

**Texto en negrita**

*Texto en cursiva*

- La negrita suele servir para conceptos clave.
- La cursiva funciona bien para terminos, enfasis suave o palabras extranjeras.

---

# Codigo en linea y bloques

Cuando escribes sobre programacion conviene diferenciar claramente el texto normal del codigo.

```markdown
Usa `print("Hola mundo")` para imprimir en pantalla.
```

Usa `print("Hola mundo")` para imprimir en pantalla.

```markdown
```python
for i in range(3):
    print(i)
```
```

```python
for i in range(3):
    print(i)
```

- El codigo en linea sirve para nombres de funciones, comandos o variables.
- Los bloques sirven para ejemplos completos.
- Si indicas el lenguaje, muchos editores activan resaltado de sintaxis.

---

# Listas

Las listas sirven para enumerar ideas, pasos o elementos relacionados sin escribir un parrafo largo.

```markdown
- Elemento 1
- Elemento 2
- Elemento 3

1. Paso 1
2. Paso 2
3. Paso 3
```

- Elemento 1
- Elemento 2
- Elemento 3

1. Paso 1
2. Paso 2
3. Paso 3

- Usa listas no ordenadas cuando el orden no importe.
- Usa listas numeradas cuando haya una secuencia real.

---

# Listas de tareas

Cuando necesitas seguir pendientes o mostrar progreso, las listas de tareas son mas utiles que una lista normal.

```markdown
- [x] Instalar dependencias
- [x] Crear notebook
- [ ] Documentar resultados
```

- [x] Instalar dependencias
- [x] Crear notebook
- [ ] Documentar resultados

- En GitHub suelen ser especialmente utiles en issues y PR.
- Tambien sirven para dividir una practica en pasos pequenos.

---

# Enlaces e imagenes

Markdown permite enlazar recursos externos y mostrar imagenes sin salir del flujo de escritura.

```markdown
[Pagina de Kaggle](https://www.kaggle.com/)

![Imagen](https://images.hdqwalls.com/wallpapers/anime-girl-living-in-fantasy-mi.jpg)
```

[Pagina de Kaggle](https://www.kaggle.com/)

![Imagen](https://images.hdqwalls.com/wallpapers/anime-girl-living-in-fantasy-mi.jpg)

- Un enlace usa texto entre corchetes y la URL entre parentesis.
- Una imagen usa la misma idea, pero con `!` al inicio.
- Si una imagen es muy grande, a veces conviene solo dejar el enlace.

---

# Citas

Las citas sirven para destacar una definicion, una observacion importante o una nota textual.

```markdown
> Esto es una cita de texto.
```

> Esto es una cita de texto.

- Funcionan bien para recordatorios, conclusiones o frases de otra fuente.
- No abuses de ellas: si todo es destacado, nada destaca.

---

# Tablas

Las tablas ayudan cuando quieres comparar datos pequenos de forma ordenada.

```markdown
| Nombre  | Rol       | Nota |
|---------|-----------|------|
| Ana     | Estudiante| 95   |
| Luis    | Tutor     | 88   |
| Marta   | Estudiante| 91   |
```

| Nombre  | Rol        | Nota |
|---------|------------|------|
| Ana     | Estudiante | 95   |
| Luis    | Tutor      | 88   |
| Marta   | Estudiante | 91   |

- Son utiles para comparar pocas filas.
- Si la tabla empieza a crecer mucho, suele ser mejor usar un archivo aparte o una hoja de calculo.

---

# Formulas con LaTeX

En notebooks y documentos tecnicos es comun mezclar Markdown con formulas matematicas.

```markdown
$y = mx + b$

$y = x^2$

$y = f\left(\sum_{i=1}^{n} (w_i \cdot x_i) + b\right)$
```

$y = mx + b$

$y = x^2$

$y = f\left(\sum_{i=1}^{n} (w_i \cdot x_i) + b\right)$

- Esto es especialmente util en IA, algebra lineal, estadistica y optimizacion.
- Si la formula es corta, puede ir en linea; si es mas importante, dejala sola.

---

# Buenas practicas

- Usa Markdown para aclarar ideas, no para decorar de mas.
- Manten una jerarquia clara de encabezados.
- Mezcla texto y codigo de forma equilibrada: explica antes o despues del bloque.
- Si el documento va a leerse rapido, prioriza ejemplos pequenos y claros.

---

# Ejercicios sugeridos

1. Escribe un documento corto con un titulo principal y al menos tres subtitulos.
2. Crea una lista no ordenada y otra numerada sobre pasos de un laboratorio.
3. Inserta una linea de codigo en linea y luego un bloque de codigo Python.
4. Agrega una lista de tareas con al menos cuatro pendientes.
5. Crea un enlace a GitHub o Kaggle y agrega una imagen publica.
6. Escribe una cita con una observacion importante sobre buenas practicas de programacion.
7. Construye una tabla con nombres, temas y calificaciones ficticias.
8. Escribe dos formulas en LaTeX: una lineal y otra relacionada con redes neuronales.

---
