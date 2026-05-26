---
theme: default
class:
  - invert
  - leap
marp: true
lang: es-ES
---

# Inteligencia Artificial

## Guia de Gradio

[Documentación oficial de Gradio](https://www.gradio.app/docs/)

### Yoel Andeyci Pilier Martinez

#### [yapmartinez@oymas.edu.do](mailto:yapmartinez@oymas.edu.do)

---

# Objetivos

- Entender cuando conviene usar `Interface` y cuando `Blocks`.
- Crear interfaces web simples sobre funciones de Python.
- Conectar componentes, eventos y salidas utiles para demos.
- Transformar ejercicios del curso en prototipos interactivos.

---

# Antes de empezar

Si ya tienes una funcion en Python, piensa primero que entra y que sale. En Gradio casi todo se vuelve mas claro cuando esa parte esta bien definida desde el principio.

- Usa `Interface` para una sola tarea directa.
- Usa `Blocks` cuando necesites ordenar mejor la app.
- Elige componentes que coincidan con el tipo de dato real.

---

# Que es Gradio

Gradio es una libreria de Python para construir interfaces web sobre funciones, APIs o modelos de machine learning sin escribir un frontend completo.

- Ideal para demos rapidas.
- Funciona bien en scripts, notebooks y prototipos.
- La documentacion oficial recomienda Python 3.10 o superior.

---

# Instalacion e importacion

```bash
pip install --upgrade gradio
```

```python
import gradio as gr
```

- El alias `gr` es la convencion mas usada.
- La app corre localmente y levanta un servidor accesible desde el navegador.

---

# Primera interfaz con `Interface`

`Interface` es util cuando una sola funcion recibe entradas y devuelve una salida clara.

```python
import gradio as gr

def greet(name, intensity):
    return "Hola, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
    api_name="predict"
)

demo.launch()
```

- Gradio genera la interfaz a partir de la firma de la funcion y los componentes elegidos.

---

# Anatomia de `Interface`

```python
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
    title="Saludador",
    description="Demo simple",
    examples=[["Ada", 1], ["PyTorch", 2]],
    live=False
)
```

- `fn`: funcion a ejecutar.
- `inputs` y `outputs`: componentes de entrada y salida.
- `examples`: ejemplos listos para probar.
- `title` y `description`: contexto para el usuario.
- `live=False` evita ejecutar en cada cambio; es util cuando el calculo no es inmediato.

---

# Componentes comunes

```python
texto = gr.Textbox(label="Prompt")
numero = gr.Number(label="Valor")
opcion = gr.Radio(["Suma", "Resta"], label="Operacion")
imagen = gr.Image(label="Entrada")
salida = gr.Textbox(label="Resultado")
```

- Usa el componente segun el tipo de dato que espera tu funcion.
- El numero de entradas debe coincidir con los parametros de la funcion.
- Elegir un buen componente reduce errores de uso desde la interfaz.

---

# Interfaces con `Blocks`

`Blocks` da mas control de layout y flujo de eventos. Es mejor cuando la app ya no cabe comodamente en una sola funcion simple.

```python
import gradio as gr

def calcular(num1, num2, operacion):
    if operacion == "Suma":
        return num1 + num2
    return num1 - num2

with gr.Blocks() as app:
    with gr.Row():
        num1 = gr.Number(label="Numero 1")
        num2 = gr.Number(label="Numero 2")
    operacion = gr.Radio(["Suma", "Resta"], label="Operacion")
    resultado = gr.Textbox(label="Resultado")
    boton = gr.Button("Calcular")
    boton.click(calcular, inputs=[num1, num2, operacion], outputs=resultado)

app.launch()
```

---

# Eventos y flujo de trabajo

- `.click(...)` conecta botones con funciones.
- `.change(...)` reacciona al cambio de un componente.
- `Blocks` da mas control de layout que `Interface`.
- `Interface` es mejor para demos pequenas; `Blocks`, para apps mas completas.

---

# Ejecucion local y recarga

```bash
python app.py
```

```bash
gradio app.py
```

- `python app.py` ejecuta la app normalmente.
- `gradio app.py` permite recarga rapida durante desarrollo.
- En notebooks, la interfaz puede mostrarse incrustada.

---

# Casos de uso en IA

- Clasificadores de texto o imagen.
- Chatbots simples.
- Interfaces para generar imagenes.
- Formularios para probar hiperparametros.
- Dashboards pequenos para mostrar predicciones.

---

# Buenas practicas

- Empieza con `Interface` si la demo es pequena; cambia a `Blocks` solo cuando haga falta mas control.
- Usa `title`, `description` y `examples` para que otra persona entienda la demo sin ayuda.
- Valida bien tipos y rangos de entrada en la funcion de Python.
- Manten la interfaz enfocada en una tarea concreta.

---

# Ejercicios sugeridos

1. Crea una app con `Interface` que convierta Celsius a Fahrenheit.
2. Agrega `title`, `description` y `examples` a esa app.
3. Haz una app que reciba dos numeros y devuelva suma, resta y multiplicacion.
4. Cambia el componente de salida por `Label` o `JSON` cuando tenga sentido.
5. Construye una calculadora con `Blocks`, `Row` y `Button`.
6. Crea una interfaz que reciba texto y cuente palabras, vocales y longitud.
7. Haz una app para subir una imagen y devolver su ancho y alto.
8. Integra una funcion de Python del curso y conviertela en demo con Gradio.

---
