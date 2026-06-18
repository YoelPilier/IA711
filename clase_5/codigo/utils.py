import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


def show_pixels(image):
    """
    Muestra los valores de intensidad de cada píxel.
    """

    img = image.convert("L")
    pixels = np.array(img)

    return (
        pd.DataFrame(pixels)
        .style.set_properties(**{"font-size": "6pt"})
        .background_gradient(cmap="Greys")
    )


def plot_confusion_matrix(Y_hat, y):
    """
    Y_hat: Tensor de tamaño (N, C) con logits o probabilidades.
    y: Tensor de tamaño (N,) con las etiquetas verdaderas.
    """

    # Predicciones
    y_pred = torch.argmax(Y_hat, dim=1)

    # Número de clases
    num_classes = max(y.max(), y_pred.max()).item() + 1

    # Matriz de confusión
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y, y_pred):
        cm[t, p] += 1

    cm = cm.numpy()

    # Mostrar
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm)

    plt.colorbar(im)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de Confusión")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    # Escribir los valores
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def show_batch(images, labels=None, cols=4, size=(8, 8)):
    """
    Muestra un batch de imágenes.

    Parámetros
    ----------
    images : Tensor
        Tensor de forma (B, C, H, W) o (B, H, W).
    labels : list | Tensor, opcional
        Etiquetas de cada imagen.
    cols : int
        Número de columnas.
    size : tuple
        Tamaño de la figura.
    """

    if not torch.is_tensor(images):
        raise TypeError("images debe ser un Tensor de PyTorch.")

    B = images.shape[0]
    rows = math.ceil(B / cols)

    fig, axes = plt.subplots(rows, cols, figsize=size)
    axes = axes.flatten() if B > 1 else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= B:
            continue

        img = images[i].detach().cpu()

        if img.ndim == 3:
            # (C,H,W) -> (H,W,C)
            img = img.permute(1, 2, 0)

            # Si tiene un solo canal
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
        else:
            # (H,W)
            ax.imshow(img, cmap="gray")

        if labels is not None:
            ax.set_title(str(labels[i]))

    plt.tight_layout()
    plt.show()
