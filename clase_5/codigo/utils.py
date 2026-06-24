import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from pathlib import Path


def plot_confusion_matrix(Y_hat, y):
    """
    Y_hat: Tensor de tamaño (N, C) con logits/probabilidades, o (N,) / (N, 1) para clasificación binaria.
    y: Tensor de tamaño (N,) con las etiquetas verdaderas.
    """
    # Asegurar que trabajamos en CPU y sin gradientes para graficar
    Y_hat = Y_hat.detach().cpu()
    y = y.detach().cpu().long()

    # --- CONTROL DE DIMENSIONES (Hacerla genérica) ---
    # Si viene con forma (N, 1), lo aplanamos a (N,)
    if Y_hat.ndim == 2 and Y_hat.shape[1] == 1:
        Y_hat = Y_hat.squeeze(1)

    if Y_hat.ndim == 1:
        # Clasificación binaria (1 salida continua)
        # Si usas Sigmoide usa > 0.5. Si usas logits directos sin activar, usa > 0.
        # Aquí asumimos > 0 (ideal para logits)
        y_pred = (Y_hat > 0).long()
        num_classes = 2
    else:
        # Clasificación multiclase (C salidas)
        y_pred = torch.argmax(Y_hat, dim=1)
        num_classes = max(y.max(), y_pred.max()).item() + 1
    # -------------------------------------------------

    # Matriz de confusión
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y, y_pred):
        # Control por si alguna etiqueta real excede el tamaño esperado
        if t < num_classes and p < num_classes:
            cm[t, p] += 1

    cm = cm.numpy()

    # Mostrar gráfico
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")  # Un mapa de color azul suele verse más limpio

    plt.colorbar(im)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de Confusión")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    # Escribir los valores dentro de las celdas
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


class ModelMonitor:
    """
    Monitor didáctico/diagnóstico para modelos PyTorch.

    Registra por capa:
    - Activaciones: mean, std, min, max, abs_mean, abs_max, zero_ratio, alive_ratio.
    - Gradientes: mean, std, min, max, abs_mean, abs_max, zero_ratio, alive_ratio.
    - Muestras de activaciones para histogramas.
    - Heatmaps de gradientes adaptados a cualquier shape.
    - Batch de entrada como imagen RGB cuando sea posible.

    Uso típico:

        monitor = ModelMonitor(model, save_dir="debug_step")

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()

        monitor.step()
        suspicious = monitor.report(show=True, save=True)

        monitor.close()
    """

    def __init__(
        self,
        model,
        save_dir="monitor_logs",
        module_filter=None,
        max_layers=None,
        capture_input=True,
        store_heatmaps=True,
        device_to_cpu=True,
        max_hist_values=4096,
        include_activation_modules=True,
        anomaly_thresholds=None,
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.module_filter = module_filter
        self.max_layers = max_layers
        self.capture_input = capture_input
        self.store_heatmaps = store_heatmaps
        self.device_to_cpu = device_to_cpu
        self.max_hist_values = max_hist_values
        self.include_activation_modules = include_activation_modules

        self.thresholds = {
            "activation_dead_std": 1e-7,
            "activation_dead_abs_mean": 1e-7,
            "activation_zero_ratio": 0.995,
            "activation_near_zero_ratio": 0.999,
            "activation_exploding_abs_max": 1e3,
            "activation_exploding_std": 1e2,
            "gradient_vanishing_abs_mean": 1e-10,
            "gradient_zero_ratio": 0.995,
            "gradient_near_zero_ratio": 0.999,
            "gradient_exploding_abs_max": 1e3,
            "gradient_exploding_std": 1e2,
            "near_zero_epsilon": 1e-12,
        }

        if anomaly_thresholds is not None:
            self.thresholds.update(anomaly_thresholds)

        self.handles = []
        self.step_id = 0
        self.layer_order = []
        self.layer_to_index = {}

        self.current_activations = {}
        self.current_gradients = {}
        self.current_grad_heatmaps = {}
        self.current_activation_samples = {}

        self.last_step_grad_heatmaps = {}
        self.last_step_activation_samples = {}
        self.last_input_batch = None

        self.activation_history = []
        self.gradient_history = []
        self.suspicious_history = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _is_leaf_module(self, module):
        return len(list(module.children())) == 0

    def _default_filter(self, name, module):
        base_types = (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.Linear,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.GroupNorm,
            torch.nn.LayerNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.MultiheadAttention,
            torch.nn.Embedding,
        )

        activation_types = (
            torch.nn.ReLU,
            torch.nn.LeakyReLU,
            torch.nn.ELU,
            torch.nn.SELU,
            torch.nn.SiLU,
            torch.nn.GELU,
            torch.nn.Tanh,
            torch.nn.Sigmoid,
            torch.nn.Softmax,
            torch.nn.LogSoftmax,
        )

        if self.include_activation_modules:
            useful_types = base_types + activation_types
        else:
            useful_types = base_types

        return isinstance(module, useful_types)

    def _tensor_from_output(self, output):
        if torch.is_tensor(output):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        if isinstance(output, dict):
            for item in output.values():
                if torch.is_tensor(item):
                    return item

        return None

    def _detach(self, x):
        x = x.detach()

        if self.device_to_cpu:
            x = x.cpu()

        return x

    def _register_hooks(self):
        if self.capture_input:
            handle = self.model.register_forward_pre_hook(self._input_hook)
            self.handles.append(handle)

        selected_modules = []

        for name, module in self.model.named_modules():
            if name == "":
                continue

            is_mha = isinstance(module, torch.nn.MultiheadAttention)

            if not self._is_leaf_module(module) and not is_mha:
                continue

            if self.module_filter is not None:
                keep = self.module_filter(name, module)
            else:
                keep = self._default_filter(name, module)

            if keep:
                selected_modules.append((name, module))

        if self.max_layers is not None:
            selected_modules = selected_modules[: self.max_layers]

        self.layer_order = [name for name, _ in selected_modules]
        self.layer_to_index = {name: i for i, name in enumerate(self.layer_order)}

        for name, module in selected_modules:
            handle = module.register_forward_hook(self._make_forward_hook(name))
            self.handles.append(handle)

    def _input_hook(self, module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            self.last_input_batch = self._detach(inputs[0])

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            out = self._tensor_from_output(output)

            if out is None:
                return

            stats = self._safe_tensor_stats(out)

            stats.update(
                {
                    "step": self.step_id,
                    "layer": name,
                    "layer_index": self.layer_to_index.get(name, -1),
                    "shape": tuple(out.shape),
                    "dtype": str(out.dtype),
                    "module_type": module.__class__.__name__,
                    "requires_grad": bool(out.requires_grad),
                }
            )

            self.current_activations[name] = stats
            self.current_activation_samples[name] = self._sample_tensor_values(out)

            if out.requires_grad:
                out.register_hook(self._make_gradient_hook(name))

        return hook

    def _make_gradient_hook(self, name):
        def grad_hook(grad):
            if grad is None:
                return

            stats = self._safe_tensor_stats(grad)

            stats.update(
                {
                    "step": self.step_id,
                    "layer": name,
                    "layer_index": self.layer_to_index.get(name, -1),
                    "shape": tuple(grad.shape),
                    "dtype": str(grad.dtype),
                }
            )

            self.current_gradients[name] = stats

            if self.store_heatmaps:
                self.current_grad_heatmaps[name] = self._make_grad_heatmap(grad)

        return grad_hook

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    def _to_real_float(self, tensor):
        x = tensor.detach()

        if torch.is_complex(x):
            x = x.abs()
        else:
            x = x.float()

        return x

    def _safe_tensor_stats(self, tensor):
        with torch.no_grad():
            x = self._to_real_float(tensor)

            if self.device_to_cpu:
                x = x.cpu()

            total = x.numel()

            if total == 0:
                return {
                    "numel": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "abs_mean": np.nan,
                    "abs_max": np.nan,
                    "zero_ratio": np.nan,
                    "near_zero_ratio": np.nan,
                    "alive_ratio": np.nan,
                    "has_nan": False,
                    "has_inf": False,
                }

            has_nan = bool(torch.isnan(x).any().item())
            has_inf = bool(torch.isinf(x).any().item())

            finite_mask = torch.isfinite(x)
            finite_x = x[finite_mask]

            zero_ratio = (x == 0).float().mean().item()

            eps = self.thresholds["near_zero_epsilon"]
            near_zero_ratio = (x.abs() < eps).float().mean().item()
            alive_ratio = 1.0 - near_zero_ratio

            if finite_x.numel() == 0:
                mean = np.nan
                std = np.nan
                min_val = np.nan
                max_val = np.nan
                abs_mean = np.nan
                abs_max = np.nan
            else:
                mean = finite_x.mean().item()
                std = finite_x.std(unbiased=False).item()
                min_val = finite_x.min().item()
                max_val = finite_x.max().item()
                abs_mean = finite_x.abs().mean().item()
                abs_max = finite_x.abs().max().item()

            return {
                "numel": int(total),
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "abs_mean": abs_mean,
                "abs_max": abs_max,
                "zero_ratio": zero_ratio,
                "near_zero_ratio": near_zero_ratio,
                "alive_ratio": alive_ratio,
                "has_nan": has_nan,
                "has_inf": has_inf,
            }

    def _sample_tensor_values(self, tensor):
        with torch.no_grad():
            x = self._to_real_float(tensor).detach().cpu().flatten()
            x = x[torch.isfinite(x)]

            if x.numel() == 0:
                return np.array([], dtype=np.float32)

            if x.numel() > self.max_hist_values:
                idx = torch.linspace(
                    0,
                    x.numel() - 1,
                    self.max_hist_values,
                ).long()
                x = x[idx]

            return x.numpy()

    # ------------------------------------------------------------------
    # Heatmaps universales
    # ------------------------------------------------------------------

    def _make_grad_heatmap(self, grad):
        with torch.no_grad():
            g = self._to_real_float(grad).abs()

            if self.device_to_cpu:
                g = g.cpu()

            if g.ndim == 0:
                hm = g.reshape(1, 1)
                kind = "scalar_0d"

            elif g.ndim == 1:
                hm = g.unsqueeze(0)
                kind = "vector_1d"

            elif g.ndim == 2:
                hm = g.mean(dim=0, keepdim=True)
                kind = "vector_2d"

            elif g.ndim == 3:
                hm = g.mean(dim=0)
                kind = "sequence_or_conv1d_3d"

            elif g.ndim == 4:
                hm = g.mean(dim=(0, 1))
                kind = "spatial_4d"

            else:
                H, W = g.shape[-2], g.shape[-1]
                hm = g.reshape(-1, H, W).mean(dim=0)
                kind = f"collapsed_{g.ndim}d"

            return {
                "heatmap": hm,
                "grad_shape": tuple(g.shape),
                "map_shape": tuple(hm.shape),
                "kind": kind,
            }

    # ------------------------------------------------------------------
    # Step y DataFrames
    # ------------------------------------------------------------------

    def step(self):
        self.activation_history.extend(self.current_activations.values())
        self.gradient_history.extend(self.current_gradients.values())

        self.last_step_activation_samples = dict(self.current_activation_samples)
        self.last_step_grad_heatmaps = dict(self.current_grad_heatmaps)

        self.current_activations = {}
        self.current_gradients = {}
        self.current_grad_heatmaps = {}
        self.current_activation_samples = {}

        self.step_id += 1

    def clear_current(self):
        self.current_activations = {}
        self.current_gradients = {}
        self.current_grad_heatmaps = {}
        self.current_activation_samples = {}

    def activations_df(self):
        return pd.DataFrame(self.activation_history)

    def gradients_df(self):
        return pd.DataFrame(self.gradient_history)

    def suspicious_df(self):
        return pd.DataFrame(self.suspicious_history)

    def _last_step_df(self, df):
        if df.empty:
            return df

        return df[df["step"] == df["step"].max()].copy()

    def _sort_df_by_forward_order(self, df):
        if df.empty:
            return df

        if "layer_index" in df.columns:
            return df.sort_values("layer_index")

        return df.sort_values("layer")

    # ------------------------------------------------------------------
    # Activaciones: gráficos didácticos
    # ------------------------------------------------------------------

    def plot_activation_stats(self, show=True, save=True, last_only=True):
        df = self.activations_df()

        if df.empty:
            print("No hay activaciones registradas.")
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, df["mean"], marker="o", label="mean")
        ax.plot(x, df["std"], marker="o", label="std")
        ax.plot(x, df["abs_mean"], marker="o", label="abs mean")
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Activaciones por capa")
        ax.set_ylabel("Valor")
        ax.legend()

        plt.tight_layout()

        if save:
            path = self.save_dir / f"activation_stats_step_{int(df['step'].max())}.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_activation_std_flow(
        self,
        show=True,
        save=True,
        last_only=True,
        log_scale=False,
    ):
        """
        Gráfico principal para enseñar inicialización:
        muestra cómo la std cambia capa por capa.
        """
        df = self.activations_df()

        if df.empty:
            print("No hay activaciones registradas.")
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))
        y = df["std"].astype(float).to_numpy()

        if log_scale:
            y = np.clip(y, 1e-20, None)

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, y, marker="o", label="activation std")

        if log_scale:
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Flujo de señal: std de activaciones")
        ax.set_ylabel("Std")
        ax.legend()

        plt.tight_layout()

        if save:
            path = (
                self.save_dir / f"activation_std_flow_step_{int(df['step'].max())}.png"
            )
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_activation_alive_ratio(
        self,
        show=True,
        save=True,
        last_only=True,
    ):
        """
        Muestra porcentaje de activaciones vivas y ceros.
        Muy útil para explicar ReLU muertas y saturación.
        """
        df = self.activations_df()

        if df.empty:
            print("No hay activaciones registradas.")
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))
        alive_pct = df["alive_ratio"].astype(float).to_numpy() * 100
        zero_pct = df["zero_ratio"].astype(float).to_numpy() * 100

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, alive_pct, marker="o", label="alive %")
        ax.plot(x, zero_pct, marker="o", label="zero %")
        ax.set_ylim(-2, 102)

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Activaciones vivas vs ceros por capa")
        ax.set_ylabel("Porcentaje")
        ax.legend()

        plt.tight_layout()

        if save:
            path = (
                self.save_dir
                / f"activation_alive_zero_step_{int(df['step'].max())}.png"
            )
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_activation_histograms(
        self,
        max_layers=8,
        bins=50,
        show=True,
        save=True,
    ):
        """
        Histogramas de activaciones del último step.
        Esta es la visualización más clara para explicar saturación.
        """
        samples = self.last_step_activation_samples or self.current_activation_samples

        if not samples:
            print("No hay muestras de activaciones para histogramas.")
            return

        items = [(name, samples[name]) for name in self.layer_order if name in samples]

        items = items[:max_layers]

        if not items:
            print("No hay capas disponibles para histogramas.")
            return

        cols = 2
        rows = math.ceil(len(items) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, (layer, values) in zip(axes, items):
            ax.axis("on")

            if values.size == 0:
                ax.set_title(f"{layer}\nsin valores finitos")
                continue

            ax.hist(values, bins=bins)
            ax.axvline(
                np.mean(values),
                linestyle="--",
                linewidth=1,
                label="mean",
            )

            ax.set_title(
                f"{layer}\nmean={np.mean(values):.3g}, std={np.std(values):.3g}"
            )
            ax.set_xlabel("Valor")
            ax.set_ylabel("Frecuencia")
            ax.legend(fontsize=8)

        plt.tight_layout()

        if save:
            path = (
                self.save_dir
                / f"activation_histograms_step_{max(0, self.step_id - 1)}.png"
            )
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_activation_min_max(self, show=True, save=True, last_only=True):
        df = self.activations_df()

        if df.empty:
            print("No hay activaciones registradas.")
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, df["min"], marker="o", label="min")
        ax.plot(x, df["max"], marker="o", label="max")
        ax.plot(x, df["abs_max"], marker="o", label="abs max")
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Rango de activaciones por capa")
        ax.set_ylabel("Valor")
        ax.legend()

        plt.tight_layout()

        if save:
            path = (
                self.save_dir / f"activation_min_max_step_{int(df['step'].max())}.png"
            )
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Gradientes: gráficos didácticos
    # ------------------------------------------------------------------

    def plot_gradient_stats(self, show=True, save=True, last_only=True):
        df = self.gradients_df()

        if df.empty:
            print(
                "No hay gradientes registrados. ¿Llamaste loss.backward() antes de monitor.step()?"
            )
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, df["abs_mean"], marker="o", label="grad abs mean")
        ax.plot(x, df["std"], marker="o", label="grad std")
        ax.plot(x, df["abs_max"], marker="o", label="grad abs max")
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Gradientes por capa")
        ax.set_ylabel("Magnitud")
        ax.legend()

        plt.tight_layout()

        if save:
            path = self.save_dir / f"gradient_stats_step_{int(df['step'].max())}.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_gradient_abs_mean_flow(
        self,
        show=True,
        save=True,
        last_only=True,
        log_scale=True,
    ):
        """
        Gráfico principal para explicar vanishing/exploding gradients.
        """
        df = self.gradients_df()

        if df.empty:
            print(
                "No hay gradientes registrados. ¿Llamaste loss.backward() antes de monitor.step()?"
            )
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))
        y = df["abs_mean"].astype(float).to_numpy()

        if log_scale:
            y = np.clip(y, 1e-20, None)

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, y, marker="o", label="grad abs mean")

        if log_scale:
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Flujo de gradientes: magnitud media")
        ax.set_ylabel("Grad abs mean")
        ax.legend()

        plt.tight_layout()

        if save:
            path = (
                self.save_dir
                / f"gradient_abs_mean_flow_step_{int(df['step'].max())}.png"
            )
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_gradient_min_max(self, show=True, save=True, last_only=True):
        df = self.gradients_df()

        if df.empty:
            print(
                "No hay gradientes registrados. ¿Llamaste loss.backward() antes de monitor.step()?"
            )
            return

        if last_only:
            df = self._last_step_df(df)

        df = self._sort_df_by_forward_order(df)

        x = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))

        ax.plot(x, df["min"], marker="o", label="grad min")
        ax.plot(x, df["max"], marker="o", label="grad max")
        ax.plot(x, df["abs_max"], marker="o", label="grad abs max")
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=75, ha="right")
        ax.set_title("Rango de gradientes por capa")
        ax.set_ylabel("Valor")
        ax.legend()

        plt.tight_layout()

        if save:
            path = self.save_dir / f"gradient_min_max_step_{int(df['step'].max())}.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_gradient_heatmaps(
        self,
        max_maps=8,
        show=True,
        save=True,
        square_cells=True,
        show_values=True,
        max_values_to_show=100,
        vector_to_square=True,
        max_display_size=64,
    ):
        """
        Heatmaps de gradientes como cuadritos tipo matriz de confusión.
        """
        items = self.last_step_grad_heatmaps or self.current_grad_heatmaps

        if not items:
            print("No hay heatmaps de gradientes disponibles.")
            return

        def resize_map_if_needed(hm, max_size):
            H, W = hm.shape

            if H <= max_size and W <= max_size:
                return hm

            hm_4d = hm.unsqueeze(0).unsqueeze(0)

            hm_small = torch.nn.functional.interpolate(
                hm_4d,
                size=(min(H, max_size), min(W, max_size)),
                mode="nearest",
            )

            return hm_small.squeeze(0).squeeze(0)

        def vector_as_square(hm):
            if hm.ndim != 2:
                hm = hm.flatten().unsqueeze(0)

            if hm.shape[0] != 1:
                return hm

            vec = hm.flatten()
            side = math.ceil(math.sqrt(vec.numel()))
            total = side * side

            if total > vec.numel():
                pad = torch.full(
                    (total - vec.numel(),),
                    float("nan"),
                    dtype=vec.dtype,
                )
                vec = torch.cat([vec, pad], dim=0)

            return vec.reshape(side, side)

        def format_number(value):
            if not np.isfinite(value):
                return ""

            if value == 0:
                return "0"

            if abs(value) < 1e-3 or abs(value) > 1e3:
                return f"{value:.1e}"

            return f"{value:.3f}"

        items = list(items.items())[:max_maps]

        cols = 2
        rows = math.ceil(len(items) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, (layer, info) in zip(axes, items):
            hm = info["heatmap"].detach().cpu().float()
            grad_shape = info["grad_shape"]
            map_shape = info["map_shape"]
            kind = info["kind"]

            if hm.ndim != 2:
                hm = hm.flatten().unsqueeze(0)

            if vector_to_square and hm.shape[0] == 1 and hm.shape[1] > 1:
                hm = vector_as_square(hm)
                visual_kind = f"{kind} -> square_view"
            else:
                visual_kind = kind

            hm = resize_map_if_needed(hm, max_display_size)

            hm_np = hm.numpy()
            H, W = hm_np.shape
            masked_hm = np.ma.masked_invalid(hm_np)

            im = ax.imshow(
                masked_hm,
                cmap="inferno",
                interpolation="nearest",
                aspect="equal" if square_cells else "auto",
            )

            if H <= 32 and W <= 32:
                ax.set_xticks(np.arange(W))
                ax.set_yticks(np.arange(H))
                ax.tick_params(axis="both", which="major", labelsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, H, 1), minor=True)

            ax.grid(
                which="minor",
                linewidth=0.5,
                color="white",
                alpha=0.35,
            )

            ax.tick_params(which="minor", bottom=False, left=False)

            if show_values and (H * W) <= max_values_to_show:
                finite_values = hm_np[np.isfinite(hm_np)]

                if finite_values.size > 0:
                    threshold = finite_values.max() / 2.0
                else:
                    threshold = 0.0

                for i in range(H):
                    for j in range(W):
                        value = hm_np[i, j]

                        if not np.isfinite(value):
                            continue

                        ax.text(
                            j,
                            i,
                            format_number(value),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="black" if value > threshold else "white",
                        )

            ax.set_title(
                f"{layer}\n"
                f"grad={grad_shape} | map={map_shape} | visual={tuple(hm.shape)}\n"
                f"{visual_kind}"
            )

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save:
            step_for_name = max(0, self.step_id - 1)
            path = self.save_dir / f"gradient_heatmaps_grid_step_{step_for_name}.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------

    def plot_activation_history(self, show=True, save=True):
        df = self.activations_df()

        if df.empty:
            print("No hay activaciones registradas.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        for layer, group in df.groupby("layer", sort=False):
            ax.plot(group["step"], group["std"], label=layer)

        ax.set_title("Historial de std de activaciones")
        ax.set_xlabel("Step")
        ax.set_ylabel("Std")
        ax.legend(fontsize=8, ncol=2)

        plt.tight_layout()

        if save:
            path = self.save_dir / "activation_std_history.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_gradient_history(self, show=True, save=True):
        df = self.gradients_df()

        if df.empty:
            print("No hay gradientes registrados.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        for layer, group in df.groupby("layer", sort=False):
            ax.plot(group["step"], group["abs_mean"], label=layer)

        ax.set_title("Historial de magnitud media de gradientes")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad abs mean")
        ax.legend(fontsize=8, ncol=2)

        plt.tight_layout()

        if save:
            path = self.save_dir / "gradient_abs_mean_history.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Detección automática
    # ------------------------------------------------------------------

    def _is_valid_number(self, value):
        try:
            return not pd.isna(value)
        except Exception:
            return False

    def _format_value(self, value):
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return "nan"

            if value == 0:
                return "0"

            if abs(value) < 1e-3 or abs(value) > 1e3:
                return f"{value:.3e}"

            return f"{value:.6f}"

        return str(value)

    def _add_issue(
        self,
        issues,
        step,
        layer,
        source,
        metric,
        value,
        severity,
        reason,
    ):
        issues.append(
            {
                "step": int(step),
                "severity": severity,
                "source": source,
                "layer": layer,
                "metric": metric,
                "value": value,
                "value_str": self._format_value(value),
                "reason": reason,
            }
        )

    def detect_suspicious_layers(
        self,
        last_only=True,
        print_warnings=True,
        save=True,
        warn_missing_grads=True,
    ):
        act_df = self.activations_df()
        grad_df = self.gradients_df()

        issues = []

        if act_df.empty and grad_df.empty:
            print("No hay estadísticas registradas todavía.")
            return pd.DataFrame()

        if last_only:
            act_df = self._last_step_df(act_df)
            grad_df = self._last_step_df(grad_df)

        t = self.thresholds

        for _, row in act_df.iterrows():
            step = row["step"]
            layer = row["layer"]

            if bool(row.get("has_nan", False)):
                self._add_issue(
                    issues,
                    step,
                    layer,
                    "activation",
                    "has_nan",
                    True,
                    "CRITICAL",
                    "La activación contiene NaN.",
                )

            if bool(row.get("has_inf", False)):
                self._add_issue(
                    issues,
                    step,
                    layer,
                    "activation",
                    "has_inf",
                    True,
                    "CRITICAL",
                    "La activación contiene Inf.",
                )

            std = row.get("std", np.nan)
            abs_mean = row.get("abs_mean", np.nan)
            abs_max = row.get("abs_max", np.nan)
            zero_ratio = row.get("zero_ratio", np.nan)
            near_zero_ratio = row.get("near_zero_ratio", np.nan)

            if self._is_valid_number(std) and self._is_valid_number(abs_mean):
                if (
                    std <= t["activation_dead_std"]
                    and abs_mean <= t["activation_dead_abs_mean"]
                ):
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "std/abs_mean",
                        max(std, abs_mean),
                        "HIGH",
                        "Posible capa muerta: std y abs_mean casi cero.",
                    )
                elif std <= t["activation_dead_std"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "std",
                        std,
                        "MEDIUM",
                        "Activación casi constante: std muy baja.",
                    )

            if self._is_valid_number(zero_ratio):
                if zero_ratio >= t["activation_zero_ratio"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "zero_ratio",
                        zero_ratio,
                        "MEDIUM",
                        "Demasiados ceros en la activación.",
                    )

            if self._is_valid_number(near_zero_ratio):
                if near_zero_ratio >= t["activation_near_zero_ratio"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "near_zero_ratio",
                        near_zero_ratio,
                        "MEDIUM",
                        "Demasiados valores casi cero en la activación.",
                    )

            if self._is_valid_number(abs_max):
                if abs_max >= t["activation_exploding_abs_max"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "abs_max",
                        abs_max,
                        "HIGH",
                        "Activación posiblemente exploding: abs_max demasiado alto.",
                    )

            if self._is_valid_number(std):
                if std >= t["activation_exploding_std"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "activation",
                        "std",
                        std,
                        "HIGH",
                        "Activación posiblemente exploding: std demasiado alta.",
                    )

        for _, row in grad_df.iterrows():
            step = row["step"]
            layer = row["layer"]

            if bool(row.get("has_nan", False)):
                self._add_issue(
                    issues,
                    step,
                    layer,
                    "gradient",
                    "has_nan",
                    True,
                    "CRITICAL",
                    "El gradiente contiene NaN.",
                )

            if bool(row.get("has_inf", False)):
                self._add_issue(
                    issues,
                    step,
                    layer,
                    "gradient",
                    "has_inf",
                    True,
                    "CRITICAL",
                    "El gradiente contiene Inf.",
                )

            std = row.get("std", np.nan)
            abs_mean = row.get("abs_mean", np.nan)
            abs_max = row.get("abs_max", np.nan)
            zero_ratio = row.get("zero_ratio", np.nan)
            near_zero_ratio = row.get("near_zero_ratio", np.nan)

            if self._is_valid_number(abs_mean):
                if abs_mean <= t["gradient_vanishing_abs_mean"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "abs_mean",
                        abs_mean,
                        "MEDIUM",
                        "Gradiente posiblemente vanishing: abs_mean casi cero.",
                    )

            if self._is_valid_number(zero_ratio):
                if zero_ratio >= t["gradient_zero_ratio"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "zero_ratio",
                        zero_ratio,
                        "MEDIUM",
                        "Demasiados ceros en el gradiente.",
                    )

            if self._is_valid_number(near_zero_ratio):
                if near_zero_ratio >= t["gradient_near_zero_ratio"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "near_zero_ratio",
                        near_zero_ratio,
                        "MEDIUM",
                        "Demasiados valores casi cero en el gradiente.",
                    )

            if self._is_valid_number(abs_max):
                if abs_max >= t["gradient_exploding_abs_max"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "abs_max",
                        abs_max,
                        "HIGH",
                        "Gradiente posiblemente exploding: abs_max demasiado alto.",
                    )

            if self._is_valid_number(std):
                if std >= t["gradient_exploding_std"]:
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "std",
                        std,
                        "HIGH",
                        "Gradiente posiblemente exploding: std demasiado alta.",
                    )

        if warn_missing_grads and not act_df.empty:
            grad_pairs = set()

            if not grad_df.empty:
                grad_pairs = set(
                    zip(
                        grad_df["step"].tolist(),
                        grad_df["layer"].tolist(),
                    )
                )

            for _, row in act_df.iterrows():
                step = row["step"]
                layer = row["layer"]

                if (
                    bool(row.get("requires_grad", False))
                    and (step, layer) not in grad_pairs
                ):
                    self._add_issue(
                        issues,
                        step,
                        layer,
                        "gradient",
                        "missing",
                        "missing",
                        "LOW",
                        "La salida requería gradiente, pero no se registró. Puede que no esté conectada a la loss.",
                    )

        issues_df = pd.DataFrame(
            issues,
            columns=[
                "step",
                "severity",
                "source",
                "layer",
                "metric",
                "value",
                "value_str",
                "reason",
            ],
        )

        if not issues_df.empty:
            severity_order = {
                "CRITICAL": 0,
                "HIGH": 1,
                "MEDIUM": 2,
                "LOW": 3,
            }

            issues_df["severity_order"] = issues_df["severity"].map(severity_order)

            issues_df = issues_df.sort_values(
                ["severity_order", "source", "layer"]
            ).drop(columns="severity_order")

        self.suspicious_history.extend(issues_df.to_dict("records"))

        if print_warnings:
            if issues_df.empty:
                print(
                    "✅ No se detectaron capas sospechosas con los umbrales actuales."
                )
            else:
                print("\n⚠️ Capas sospechosas detectadas:\n")

                cols = [
                    "severity",
                    "source",
                    "layer",
                    "metric",
                    "value_str",
                    "reason",
                ]

                print(issues_df[cols].to_string(index=False))

        if save and not issues_df.empty:
            step_for_name = int(issues_df["step"].max())
            path = self.save_dir / f"suspicious_layers_step_{step_for_name}.csv"
            issues_df.to_csv(path, index=False)

        return issues_df

    def plot_suspicious_table(
        self,
        issues_df=None,
        max_rows=25,
        show=True,
        save=True,
    ):
        if issues_df is None:
            issues_df = self.detect_suspicious_layers(
                last_only=True,
                print_warnings=False,
                save=False,
            )

        if issues_df.empty:
            print("No hay capas sospechosas para graficar.")
            return

        display_cols = [
            "severity",
            "source",
            "layer",
            "metric",
            "value_str",
            "reason",
        ]

        table_df = issues_df[display_cols].head(max_rows).copy()

        fig_height = max(3, 0.45 * len(table_df) + 1.5)

        fig, ax = plt.subplots(figsize=(14, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="left",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        ax.set_title("Capas sospechosas", pad=20)

        plt.tight_layout()

        if save:
            step_for_name = int(issues_df["step"].max())
            path = self.save_dir / f"suspicious_layers_table_step_{step_for_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Input RGB
    # ------------------------------------------------------------------

    def show_input_rgb(
        self,
        channels=(0, 1, 2),
        max_images=4,
        normalize=True,
        show=True,
        save=True,
    ):
        if self.last_input_batch is None:
            print("No hay batch de entrada capturado.")
            return

        x = self.last_input_batch

        if x.ndim != 4:
            print(
                f"El batch capturado no tiene forma (B, C, H, W). "
                f"Shape recibido: {tuple(x.shape)}"
            )
            return

        B, C, H, W = x.shape
        n = min(B, max_images)

        images = x[:n].detach().cpu().float()

        if C >= 3:
            valid_channels = [c for c in channels if c < C]

            if len(valid_channels) < 3:
                valid_channels = list(range(min(3, C)))

            if len(valid_channels) < 3:
                print(f"No se pudieron seleccionar 3 canales válidos. C={C}")
                return

            images = images[:, valid_channels[:3], :, :]

        elif C == 2:
            images = torch.cat(
                [
                    images[:, 0:1],
                    images[:, 1:2],
                    images[:, 0:1],
                ],
                dim=1,
            )

        elif C == 1:
            images = images.repeat(1, 3, 1, 1)

        else:
            print(f"Número de canales inválido: C={C}")
            return

        if normalize:
            images = self._normalize_images_for_display(images)

        cols = min(n, 4)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for i in range(n):
            img = images[i].permute(1, 2, 0).numpy()

            axes[i].imshow(img)
            axes[i].set_title(f"Input {i}")
            axes[i].axis("off")

        plt.tight_layout()

        if save:
            step_for_name = max(0, self.step_id - 1)
            path = self.save_dir / f"input_rgb_step_{step_for_name}.png"
            plt.savefig(path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _normalize_images_for_display(self, images):
        imgs = images.clone()

        for i in range(imgs.shape[0]):
            img = imgs[i]

            min_val = img.min()
            max_val = img.max()

            if (max_val - min_val) > 1e-8:
                imgs[i] = (img - min_val) / (max_val - min_val)
            else:
                imgs[i] = torch.zeros_like(img)

        return imgs.clamp(0, 1)

    # ------------------------------------------------------------------
    # Guardado y reporte
    # ------------------------------------------------------------------

    def save_tables(self):
        act_df = self.activations_df()
        grad_df = self.gradients_df()
        susp_df = self.suspicious_df()

        if not act_df.empty:
            act_df.to_csv(self.save_dir / "activations.csv", index=False)

        if not grad_df.empty:
            grad_df.to_csv(self.save_dir / "gradients.csv", index=False)

        if not susp_df.empty:
            susp_df.to_csv(self.save_dir / "suspicious_history.csv", index=False)

    def report(
        self,
        show=True,
        save=True,
        include_ranges=True,
        include_history=False,
        include_heatmaps=True,
        include_input=True,
        include_histograms=True,
        include_teaching_plots=True,
        include_suspicious=True,
        plot_suspicious=True,
    ):
        """
        Genera reporte general.

        Retorna:
            DataFrame con capas sospechosas.
        """
        self.plot_activation_stats(show=show, save=save)

        if include_teaching_plots:
            self.plot_activation_std_flow(show=show, save=save)
            self.plot_activation_alive_ratio(show=show, save=save)
            self.plot_gradient_abs_mean_flow(show=show, save=save)

        if include_histograms:
            self.plot_activation_histograms(show=show, save=save)

        if include_ranges:
            self.plot_activation_min_max(show=show, save=save)
            self.plot_gradient_min_max(show=show, save=save)

        self.plot_gradient_stats(show=show, save=save)

        if include_heatmaps:
            self.plot_gradient_heatmaps(show=show, save=save)

        if include_input:
            self.show_input_rgb(show=show, save=save)

        if include_history:
            self.plot_activation_history(show=show, save=save)
            self.plot_gradient_history(show=show, save=save)

        suspicious = None

        if include_suspicious:
            suspicious = self.detect_suspicious_layers(
                last_only=True,
                print_warnings=True,
                save=save,
            )

            if plot_suspicious and suspicious is not None and not suspicious.empty:
                self.plot_suspicious_table(
                    issues_df=suspicious,
                    show=show,
                    save=save,
                )

        if save:
            self.save_tables()

        return suspicious

    def close(self):
        for handle in self.handles:
            handle.remove()

        self.handles = []


def compare_monitors(
    monitors,
    labels=None,
    metric="activation_std",
    show=True,
    save=True,
    save_dir="monitor_comparison",
):
    """
    Compara varios ModelMonitor en una misma gráfica.

    Útil para clase:

        ReLU + Xavier vs ReLU + He vs SiLU + He

    metric puede ser:
        - "activation_std"
        - "activation_alive"
        - "gradient_abs_mean"
    """
    if isinstance(monitors, dict):
        labels = list(monitors.keys())
        monitors = list(monitors.values())

    if labels is None:
        labels = [f"run_{i}" for i in range(len(monitors))]

    rows = []

    for label, monitor in zip(labels, monitors):
        if metric in ("activation_std", "activation_alive"):
            df = monitor._last_step_df(monitor.activations_df())

            if df.empty:
                continue

            df = monitor._sort_df_by_forward_order(df)

            for _, row in df.iterrows():
                if metric == "activation_std":
                    value = row["std"]
                else:
                    value = row["alive_ratio"] * 100

                rows.append(
                    {
                        "run": label,
                        "layer": row["layer"],
                        "layer_index": row.get("layer_index", -1),
                        "value": value,
                    }
                )

        elif metric == "gradient_abs_mean":
            df = monitor._last_step_df(monitor.gradients_df())

            if df.empty:
                continue

            df = monitor._sort_df_by_forward_order(df)

            for _, row in df.iterrows():
                rows.append(
                    {
                        "run": label,
                        "layer": row["layer"],
                        "layer_index": row.get("layer_index", -1),
                        "value": row["abs_mean"],
                    }
                )

        else:
            raise ValueError(
                "metric debe ser 'activation_std', "
                "'activation_alive' o 'gradient_abs_mean'."
            )

    comp_df = pd.DataFrame(rows)

    if comp_df.empty:
        print("No hay datos para comparar.")
        return comp_df

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layers = (
        comp_df[["layer", "layer_index"]]
        .drop_duplicates()
        .sort_values("layer_index")["layer"]
        .tolist()
    )

    x = np.arange(len(layers))

    title_map = {
        "activation_std": "Comparación: std de activaciones",
        "activation_alive": "Comparación: porcentaje de activaciones vivas",
        "gradient_abs_mean": "Comparación: magnitud media de gradientes",
    }

    y_label_map = {
        "activation_std": "Std",
        "activation_alive": "Alive %",
        "gradient_abs_mean": "Grad abs mean",
    }

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.45), 5))

    for label in labels:
        run_df = comp_df[comp_df["run"] == label]

        values = []

        for layer in layers:
            match = run_df[run_df["layer"] == layer]

            if match.empty:
                values.append(np.nan)
            else:
                values.append(match["value"].iloc[0])

        if metric == "gradient_abs_mean":
            values = np.clip(np.array(values, dtype=float), 1e-20, None)

        ax.plot(x, values, marker="o", label=label)

    if metric == "gradient_abs_mean":
        ax.set_yscale("log")

    if metric == "activation_alive":
        ax.set_ylim(-2, 102)

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=75, ha="right")
    ax.set_title(title_map[metric])
    ax.set_ylabel(y_label_map[metric])
    ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save_dir / f"compare_{metric}.png", dpi=150)
        comp_df.to_csv(save_dir / f"compare_{metric}.csv", index=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return comp_df
