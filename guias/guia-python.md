---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de Python

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Repasar la sintaxis esencial de Python.
- Preparar un entorno virtual aislado para cada proyecto.
- Entender variables, colecciones, funciones y clases.
- Introducir buenas practicas utiles para laboratorios de IA.
- Cerrar con ejercicios que mezclen logica, archivos y analisis simple.

---

# Antes de empezar

Si llegaste a esta guia por una duda puntual, no hace falta leerla completa de una vez. Ve directo a la seccion que necesites, ejecuta el ejemplo mas pequeno posible y luego prueba una variacion propia.

- Cuando una idea no salga, reduce el problema.
- Cuando un tipo de dato te confunda, imprimelo.
- Cuando una sintaxis se olvide, rehace un ejemplo corto antes de seguir.

---

# Crear un entorno virtual con `venv`

Un entorno virtual aisla dependencias por proyecto. Esto evita conflictos entre versiones y facilita reproducir un laboratorio en otra maquina.

```bash
python -m venv .venv
```

Activacion segun el sistema:

```bash
source .venv/bin/activate
```

```bash
.venv\Scripts\activate
```

- Si el prompt cambia, normalmente el entorno ya esta activo.
- Para salir del entorno virtual usa `deactivate`.

---

# Instalar paquetes

Dentro del entorno virtual instala solo las librerias que necesite el proyecto. En este curso es comun trabajar con `numpy`, `pandas`, `matplotlib`, `torch` o `gradio`.

```bash
pip install numpy pandas matplotlib
```

Tambien puedes instalar desde un archivo de dependencias:

```bash
pip install -r requirements.txt
```

Verifica lo instalado con:

```bash
pip list
```

- `requirements.txt` ayuda a compartir el mismo entorno con otras personas.
- Instalar paquetes globalmente suele complicar el mantenimiento del sistema.

---

# Python

Python es un lenguaje interpretado, legible y multiparadigma. En los laboratorios del curso sirve para automatizar tareas, procesar datos, visualizar resultados y crear modelos sencillos.

```python
print("Hola, IA")
```

- La sintaxis busca ser clara: menos simbolos, mas enfasis en legibilidad.
- La indentacion es parte del lenguaje, no solo estilo visual.

---

# Variables, tipos y lectura de datos

```python
nombre = "Ada"
edad = 20
altura = 1.68
activo = True

print(type(nombre), type(edad), type(altura), type(activo))

dato = input("Ingresa un numero: ")
numero = int(dato)
print(numero * 2)
```

- Python usa tipado dinamico: el tipo se deduce del valor asignado.
- `input()` siempre devuelve texto, por eso conviene convertir a `int`, `float` u otro tipo.
- `type(...)` ayuda a inspeccionar que tipo de dato tienes realmente.

---

# Operadores y control de flujo

```python
x = 8
y = 3

print(x + y, x - y, x * y, x / y, x % y, x ** y)

if x > y:
    print("x es mayor")
elif x == y:
    print("son iguales")
else:
    print("y es mayor")

for i in range(3):
    print(i)
```

- `if`, `elif` y `else` permiten tomar decisiones.
- `for` recorre secuencias; `range(n)` genera valores desde `0` hasta `n - 1`.
- `**` calcula potencias y `%` devuelve el residuo.

---

# Funciones

Las funciones permiten reutilizar logica y dividir un problema grande en pasos pequenos y claros.

```python
def saludar(nombre="Mundo"):
    return f"Hola {nombre}"

def sumar_y_restar(a, b):
    return a + b, a - b

def suma_total(*numeros):
    return sum(numeros)

print(saludar("Python"))
print(sumar_y_restar(10, 4))
print(suma_total(1, 2, 3, 4))
```

- Un parametro con valor por defecto vuelve mas flexible la funcion.
- Una funcion puede devolver varios valores, que en realidad viajan como una tupla.
- `*numeros` agrupa una cantidad variable de argumentos.

---

# Colecciones

```python
frutas = ["manzana", "banana", "cereza"]
coordenada = (10, 20)
persona = {"nombre": "Juan", "edad": 30}

frutas[1] = "naranja"
persona["ciudad"] = "Santo Domingo"

print(frutas)
print(coordenada[0])
print(persona["nombre"])
```

- `list`: ordenada y mutable; sirve cuando necesitas agregar o cambiar elementos.
- `tuple`: ordenada e inmutable; conviene para datos que no deben alterarse.
- `dict`: guarda pares clave-valor; aparece mucho al trabajar con JSON y configuraciones.

---

# Comprensiones y utilidades

```python
pares = [x for x in range(10) if x % 2 == 0]
cuadrados = {x: x**2 for x in range(5)}

for indice, valor in enumerate(["a", "b", "c"]):
    print(indice, valor)

for letra, numero in zip(["a", "b", "c"], [1, 2, 3]):
    print(letra, numero)
```

- Las comprensiones resumen la idea de "crear una coleccion a partir de otra".
- `enumerate()` da indice y valor al mismo tiempo.
- `zip()` recorre varias secuencias en paralelo.

---

# Clases y objetos

Una clase agrupa datos y comportamiento. En IA no siempre se crean muchas clases, pero es util entenderlas para leer librerias y disenar proyectos mas grandes.

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def saludar(self):
        return f"Hola, soy {self.nombre}"

class Empleado(Persona):
    def __init__(self, nombre, edad, sueldo):
        super().__init__(nombre, edad)
        self.sueldo = sueldo
```

- `self` representa la instancia actual.
- `__init__` inicializa el objeto al crearlo.
- La herencia permite reutilizar y extender logica existente.

---

# Complejidad computacional

La complejidad describe como crece el costo de un algoritmo cuando aumenta el tamano de la entrada. No mide segundos exactos, sino tendencia de crecimiento.

```python
def buscar_lineal(datos, objetivo):
    for valor in datos:      # O(n)
        if valor == objetivo:
            return True
    return False

def pares_en_matriz(datos):
    for i in datos:          # O(n^2)
        for j in datos:
            print(i, j)
```

- `O(1)`: costo constante.
- `O(log n)`: crece lentamente al aumentar la entrada.
- `O(n)`: revisa una proporcion directa de elementos.
- `O(n^2)`: aparece mucho con bucles anidados.
- `O(2^n)`: costo exponencial; al agregar un elemento, el trabajo puede casi duplicarse.
- `O(n!)`: costo factorial; aparece cuando hay que explorar muchisimos ordenamientos o combinaciones posibles.

Las complejidades exponencial y factorial son malas porque crecen demasiado rapido. Un algoritmo que parece tolerable con entradas pequenas puede volverse impractico con un aumento pequeno de `n`.

```python
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)   # O(2^n)
```

- En `O(2^n)` cada paso puede abrir varias ramas nuevas.
- En `O(n!)` el crecimiento es peor todavia: `5! = 120`, `10! = 3628800`.
- Por eso son tan malas: el costo explota y deja de escalar muy rapido.

---

# Archivos y manejo de errores

```python
with open("salida.txt", "w", encoding="utf-8") as archivo:
    archivo.write("Hola mundo")

try:
    resultado = 10 / 0
except ZeroDivisionError:
    print("No se puede dividir entre cero")
```

- `with open(...)` cierra el archivo automaticamente al terminar.
- `try/except` permite capturar errores esperables y dar un mensaje mas claro.
- Conviene capturar errores especificos, no cualquier excepcion sin revisar.

---

# Buenas practicas

- Usa nombres descriptivos para variables y funciones.
- Separa la lectura, el procesamiento y la salida en pasos distintos.
- Evita repetir codigo: si una logica se repite, conviertela en funcion.
- Imprime o inspecciona resultados pequenos antes de pasar a ejemplos grandes.

---

# Ejercicios sugeridos

1. Crea un programa que lea nombre y edad del usuario y muestre un mensaje personalizado.
2. Escribe una funcion que reciba una lista de numeros y devuelva suma, promedio y maximo.
3. Pide cinco numeros al usuario, guardalos en una lista y muestra solo los pares.
4. Crea un diccionario con datos de tres estudiantes y recorre sus claves y valores.
5. Implementa una clase `CuentaBancaria` con metodos para depositar y retirar.
6. Lee un archivo de texto y cuenta cuantas lineas y palabras contiene.
7. Escribe un programa que intente dividir dos numeros e informe un error claro si el divisor es cero.
8. Compara con ejemplos propios una solucion `O(n)` y otra `O(n^2)` y explica cual escala mejor.

---
