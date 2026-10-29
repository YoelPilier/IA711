"""Gráficos y diagnóstico didáctico de modelos PyTorch.

El monitor mide gradientes de las salidas de cada capa, no de sus parámetros.
Los umbrales son orientativos y dependen del modelo, la pérdida y el batch.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_confusion_matrix(Y_hat, y, threshold=0.0):
    """Grafica logits (N, C) o salidas binarias (N,) / (N, 1).

    Para probabilidades binarias usa threshold=0.5; para logits, threshold=0.
    Devuelve (fig, ax) para poder personalizar o inspeccionar el gráfico.
    """
    Y_hat = torch.as_tensor(Y_hat).detach().cpu()
    y = torch.as_tensor(y).detach().cpu()
    if y.ndim != 1 or not y.numel():
        raise ValueError("y debe tener forma (N,) y contener etiquetas.")
    if Y_hat.ndim not in (1, 2) or Y_hat.shape[0] != y.numel():
        raise ValueError("Y_hat debe tener forma (N,), (N, 1) o (N, C).")
    if Y_hat.ndim == 2 and Y_hat.shape[1] == 0:
        raise ValueError("Y_hat debe tener al menos una salida por muestra.")
    if not torch.isfinite(Y_hat).all() or not torch.isfinite(y).all():
        raise ValueError("Las predicciones y etiquetas deben ser finitas.")
    if not math.isfinite(threshold):
        raise ValueError("threshold debe ser finito.")
    if (y != y.long()).any():
        raise ValueError("Las etiquetas deben ser enteros.")
    y = y.long()
    if Y_hat.ndim == 1 or Y_hat.shape[1] == 1:
        y_pred = (Y_hat.reshape(-1) > threshold).long()
        num_classes = 2
    else:
        y_pred = Y_hat.argmax(dim=1)
        num_classes = Y_hat.shape[1]
    if ((y < 0) | (y >= num_classes)).any():
        raise ValueError(f"Las etiquetas deben estar entre 0 y {num_classes - 1}.")
    cm = torch.bincount(
        y * num_classes + y_pred, minlength=num_classes**2
    ).reshape(num_classes, num_classes).numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xlabel="Predicción", ylabel="Etiqueta real", title="Matriz de confusión")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png")
    plt.show()
    return fig, ax


def show_batch(images, labels=None, cols=4, size=(8, 8)):
    """Muestra imágenes (B, C, H, W) o (B, H, W); devuelve (fig, axes)."""
    if not torch.is_tensor(images):
        raise TypeError("images debe ser un Tensor de PyTorch.")
    if images.ndim not in (3, 4) or any(s == 0 for s in images.shape):
        raise ValueError("images debe tener forma (B, C, H, W) o (B, H, W), sin ejes vacíos.")
    if not isinstance(cols, int) or cols < 1:
        raise ValueError("cols debe ser un entero positivo.")
    if images.ndim == 4 and images.shape[1] not in (1, 3, 4):
        raise ValueError("Las imágenes deben tener 1, 3 o 4 canales.")
    if labels is not None and len(labels) != len(images):
        raise ValueError("Debe haber una etiqueta por imagen.")
    fig, axes = plt.subplots(math.ceil(len(images) / cols), cols,
                             figsize=size, squeeze=False)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(images):
            continue
        img = images[i].detach().cpu()
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if labels is not None:
            label = labels[i]
            if torch.is_tensor(label):
                label = label.detach().cpu().tolist()
            ax.set_title(str(label))
    fig.tight_layout()
    plt.show()
    return fig, axes


class ModelMonitor:
    """Captura activaciones y gradientes de salida sin cambiar el entrenamiento.

    Uso::

        with ModelMonitor(model) as monitor:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            monitor.step()
            suspicious = monitor.report(show=True, save=True)

    También se puede llamar close() explícitamente. Llama step() después del
    backward de cada batch. Las llamadas repetidas a un módulo se distinguen
    con sufijos #2, #3, etc. El historial crece hasta llamar reset().
    """

    ISSUE_COLUMNS = ["step", "severity", "source", "layer", "metric",
                     "value", "value_str", "reason"]

    def __init__(self, model, save_dir="monitor_logs", module_filter=None,
                 max_layers=None, capture_input=True, store_heatmaps=True,
                 device_to_cpu=True, max_hist_values=4096,
                 include_activation_modules=True, anomaly_thresholds=None):
        if max_layers is not None and (not isinstance(max_layers, int) or max_layers < 1):
            raise ValueError("max_layers debe ser un entero positivo o None.")
        if not isinstance(max_hist_values, int) or max_hist_values < 1:
            raise ValueError("max_hist_values debe ser un entero positivo.")
        self.model = model
        self.save_dir = Path(save_dir)
        self.module_filter = module_filter
        self.max_layers = max_layers
        self.capture_input = capture_input
        self.store_heatmaps = store_heatmaps
        self.device_to_cpu = device_to_cpu
        self.max_hist_values = max_hist_values
        self.include_activation_modules = include_activation_modules
        self.thresholds = {
            "activation_dead_std": 1e-7, "activation_dead_abs_mean": 1e-7,
            "activation_zero_ratio": 0.995, "activation_near_zero_ratio": 0.999,
            "activation_exploding_abs_max": 1e3, "activation_exploding_std": 1e2,
            "gradient_vanishing_abs_mean": 1e-10, "gradient_zero_ratio": 0.995,
            "gradient_near_zero_ratio": 0.999, "gradient_exploding_abs_max": 1e3,
            "gradient_exploding_std": 1e2, "near_zero_epsilon": 1e-12,
        }
        if anomaly_thresholds is not None:
            unknown = set(anomaly_thresholds) - self.thresholds.keys()
            if unknown:
                raise ValueError(f"Umbrales desconocidos: {sorted(unknown)}")
            self.thresholds.update(anomaly_thresholds)
        for key, value in self.thresholds.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"El umbral {key} debe ser finito y no negativo.")
        if self.thresholds["near_zero_epsilon"] == 0:
            raise ValueError("near_zero_epsilon debe ser mayor que cero.")
        self.handles = []
        self._tensor_handles = []
        self.layer_order = []
        self.layer_to_index = {}
        self.reset()
        self._register_hooks()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self):
        """Libera las capturas y el historial; conserva los hooks del modelo."""
        self.clear_current()
        self.step_id = 0
        self.layer_order = []
        self.layer_to_index = {}
        self.last_step_grad_heatmaps = {}
        self.last_step_activation_samples = {}
        self.last_input_batch = None
        self.activation_history = []
        self.gradient_history = []
        self.suspicious_history = []

    def _default_filter(self, name, module):
        base = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm,
                torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d, torch.nn.MultiheadAttention, torch.nn.Embedding)
        activations = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU,
                       torch.nn.SELU, torch.nn.SiLU, torch.nn.GELU, torch.nn.Tanh,
                       torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.LogSoftmax)
        return isinstance(module, base + activations if self.include_activation_modules else base)

    def _tensor_from_output(self, output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            output = output.values()
        elif not isinstance(output, (list, tuple)):
            return None
        for item in output:
            tensor = self._tensor_from_output(item)
            if tensor is not None:
                return tensor
        return None

    def _register_hooks(self):
        if self.capture_input:
            self.handles.append(self.model.register_forward_pre_hook(self._input_hook))
        selected = []
        for name, module in self.model.named_modules():
            if any(module.children()) and not isinstance(module, torch.nn.MultiheadAttention):
                continue
            name = name or "(model)"
            keep = (self.module_filter(name, module) if self.module_filter is not None
                    else self._default_filter(name, module))
            if keep:
                selected.append((name, module))
        if self.max_layers is not None:
            selected = selected[:self.max_layers]
        for name, module in selected:
            self.handles.append(module.register_forward_hook(self._make_forward_hook(name)))

    def _input_hook(self, module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            x = inputs[0].detach()
            self.last_input_batch = (x.cpu() if self.device_to_cpu else x).clone()

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            out = self._tensor_from_output(output)
            if out is None:
                return
            count = self._call_counts.get(name, 0) + 1
            self._call_counts[name] = count
            # No sobrescribir estadísticas si el módulo se reutiliza.
            key = name if count == 1 else f"{name}#{count}"
            while key in self.current_activations:
                key += "#"
            index = len(self.current_activations)
            if key not in self.layer_to_index:
                self.layer_order.append(key)
            self.layer_to_index[key] = index
            stats = self._safe_tensor_stats(out)
            stats.update(step=self.step_id, layer=key, layer_index=index,
                         shape=tuple(out.shape), dtype=str(out.dtype),
                         module_type=module.__class__.__name__, requires_grad=out.requires_grad)
            self.current_activations[key] = stats
            self.current_activation_samples[key] = self._sample_tensor_values(out)
            if out.requires_grad:
                self._tensor_handles.append(out.register_hook(
                    self._make_gradient_hook(key, self.step_id, index)))
        return hook

    def _make_gradient_hook(self, name, step, index):
        def hook(grad):
            if grad is None:
                return
            stats = self._safe_tensor_stats(grad)
            stats.update(step=step, layer=name, layer_index=index,
                         shape=tuple(grad.shape), dtype=str(grad.dtype))
            self.current_gradients[name] = stats
            if self.store_heatmaps:
                self.current_grad_heatmaps[name] = self._make_grad_heatmap(grad)
        return hook

    @staticmethod
    def _to_real_float(tensor):
        x = tensor.detach()
        return x.abs().float() if torch.is_complex(x) else x.float()

    def _safe_tensor_stats(self, tensor):
        with torch.no_grad():
            x = self._to_real_float(tensor)
            if self.device_to_cpu:
                x = x.cpu()
            finite = torch.isfinite(x)
            values = x[finite]
            stats = dict.fromkeys(
                ("mean", "std", "min", "max", "abs_mean", "abs_max",
                 "zero_ratio", "near_zero_ratio", "alive_ratio"), np.nan)
            stats.update(numel=x.numel(), has_nan=torch.isnan(x).any().item(),
                         has_inf=torch.isinf(x).any().item())
            if x.numel():
                stats["zero_ratio"] = (x == 0).float().mean().item()
                stats["near_zero_ratio"] = (x.abs() < self.thresholds["near_zero_epsilon"]).float().mean().item()
                # Los valores no finitos no cuentan como activos.
                stats["alive_ratio"] = (finite & (x.abs() >= self.thresholds["near_zero_epsilon"])).float().mean().item()
            if values.numel():
                stats.update(mean=values.mean().item(), std=values.std(unbiased=False).item(),
                             min=values.min().item(), max=values.max().item(),
                             abs_mean=values.abs().mean().item(), abs_max=values.abs().max().item())
            return stats

    def _sample_tensor_values(self, tensor):
        x = self._to_real_float(tensor).flatten()
        # Acotar la transferencia a CPU antes de copiar el tensor completo.
        if x.numel() > self.max_hist_values:
            idx = torch.linspace(0, x.numel() - 1, self.max_hist_values,
                                 device=x.device).long()
            x = x[idx]
        x = x[torch.isfinite(x)]
        return x.cpu().numpy().copy()

    def _make_grad_heatmap(self, grad):
        g = self._to_real_float(grad).abs()
        if g.ndim == 0:
            hm, kind = g.reshape(1, 1), "scalar_0d"
        elif g.ndim == 1:
            hm, kind = g.unsqueeze(0), "vector_1d"
        elif g.ndim == 2:
            hm, kind = g.mean(dim=0, keepdim=True), "vector_2d"
        elif g.ndim == 3:
            hm, kind = g.mean(dim=0), "sequence_or_conv1d_3d"
        else:
            hm = g.reshape(-1, *g.shape[-2:]).mean(dim=0)
            kind = f"collapsed_{g.ndim}d"
        hm = hm.cpu() if self.device_to_cpu else hm
        return dict(heatmap=hm.clone(), grad_shape=tuple(g.shape),
                    map_shape=tuple(hm.shape), kind=kind)

    def clear_current(self):
        for handle in self._tensor_handles:
            handle.remove()
        self._tensor_handles.clear()
        self.current_activations = {}
        self.current_gradients = {}
        self.current_grad_heatmaps = {}
        self.current_activation_samples = {}
        self._call_counts = {}

    def step(self):
        self.activation_history.extend(self.current_activations.values())
        self.gradient_history.extend(self.current_gradients.values())
        self.last_step_activation_samples = dict(self.current_activation_samples)
        self.last_step_grad_heatmaps = dict(self.current_grad_heatmaps)
        self.clear_current()
        self.step_id += 1

    def activations_df(self):
        return pd.DataFrame(self.activation_history)

    def gradients_df(self):
        return pd.DataFrame(self.gradient_history)

    def suspicious_df(self):
        return pd.DataFrame(self.suspicious_history, columns=self.ISSUE_COLUMNS)

    def _last_step_df(self, df):
        # Usar el último paso confirmado, incluso si no produjo gradientes.
        if df.empty:
            return df.copy()
        return df[df["step"] == self.step_id - 1].copy()

    @staticmethod
    def _sort_df_by_forward_order(df):
        if df.empty:
            return df
        return df.sort_values(["step", "layer_index"], kind="stable")

    def _finish_plot(self, fig, filename, show, save):
        fig.tight_layout()
        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_stats(self, source, metrics, title, ylabel, filename,
                    show=True, save=True, last_only=True, log_scale=False,
                    percent=False):
        df = self.activations_df() if source == "activation" else self.gradients_df()
        if last_only:
            df = self._last_step_df(df)
        if df.empty:
            print(f"No hay datos de {source} para los pasos solicitados.")
            return
        df = self._sort_df_by_forward_order(df)
        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))
        x = np.arange(len(df))
        for metric, label in metrics:
            values = df[metric].to_numpy(dtype=float) * (100 if percent else 1)
            if log_scale:
                values = np.clip(values, 1e-20, None)
            ax.plot(x, values, marker="o", label=label)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.axhline(0, linestyle="--", linewidth=1)
        if percent:
            ax.set_ylim(-2, 102)
        labels = (df["layer"] if last_only else
                  df["layer"] + " (step " + df["step"].astype(str) + ")")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.set(title=title, ylabel=ylabel)
        ax.legend()
        self._finish_plot(fig, f"{filename}_step_{int(df['step'].max())}.png", show, save)

    def plot_activation_stats(self, show=True, save=True, last_only=True):
        self._plot_stats("activation", [("mean", "mean"), ("std", "std"),
                         ("abs_mean", "abs mean")], "Activaciones por capa", "Valor",
                         "activation_stats", show, save, last_only)

    def plot_activation_std_flow(self, show=True, save=True, last_only=True, log_scale=False):
        self._plot_stats("activation", [("std", "activation std")],
                         "Flujo de señal: std de activaciones", "Std",
                         "activation_std_flow", show, save, last_only, log_scale)

    def plot_activation_alive_ratio(self, show=True, save=True, last_only=True):
        """Porcentaje no cercano a cero; no es una medida de saturación."""
        self._plot_stats("activation", [("alive_ratio", "No cercanos a cero %"),
                         ("zero_ratio", "Ceros %")], "Valores no cercanos a cero por capa",
                         "Porcentaje", "activation_alive_zero", show, save, last_only,
                         percent=True)

    def plot_activation_min_max(self, show=True, save=True, last_only=True):
        self._plot_stats("activation", [("min", "min"), ("max", "max"),
                         ("abs_max", "abs max")], "Rango de activaciones por capa",
                         "Valor", "activation_min_max", show, save, last_only)

    def plot_gradient_stats(self, show=True, save=True, last_only=True):
        self._plot_stats("gradient", [("abs_mean", "grad abs mean"), ("std", "grad std"),
                         ("abs_max", "grad abs max")], "Gradientes de salida por capa",
                         "Magnitud", "gradient_stats", show, save, last_only)

    def plot_gradient_abs_mean_flow(self, show=True, save=True, last_only=True, log_scale=True):
        self._plot_stats("gradient", [("abs_mean", "grad abs mean")],
                         "Flujo de gradientes: magnitud media", "Grad abs mean",
                         "gradient_abs_mean_flow", show, save, last_only, log_scale)

    def plot_gradient_min_max(self, show=True, save=True, last_only=True):
        self._plot_stats("gradient", [("min", "min"), ("max", "max"),
                         ("abs_max", "abs max")], "Rango de gradientes por capa",
                         "Valor", "gradient_min_max", show, save, last_only)

    def plot_activation_histograms(self, max_layers=8, bins=50, show=True, save=True):
        if max_layers < 1 or bins < 1:
            raise ValueError("max_layers y bins deben ser positivos.")
        current = bool(self.current_activations)
        samples = self.current_activation_samples if current else self.last_step_activation_samples
        items = list(samples.items())[:max_layers]
        if not items:
            print("No hay muestras de activaciones para histogramas.")
            return
        fig, axes = plt.subplots(math.ceil(len(items) / 2), 2,
                                 figsize=(12, math.ceil(len(items) / 2) * 4), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (name, values) in zip(axes.ravel(), items):
            ax.axis("on")
            ax.set_title(name)
            if not values.size:
                ax.set_title(f"{name}: sin valores finitos")
                continue
            ax.hist(values, bins=bins)
            ax.axvline(values.mean(), linestyle="--", label="mean")
            ax.set(xlabel="Valor", ylabel="Frecuencia")
            ax.legend()
        step = self.step_id if current else self.step_id - 1
        self._finish_plot(fig, f"activation_histograms_step_{step}.png", show, save)

    def plot_gradient_heatmaps(self, max_maps=8, show=True, save=True,
                               square_cells=True, show_values=True,
                               max_values_to_show=100, vector_to_square=True,
                               max_display_size=64):
        if max_maps < 1 or max_display_size < 1:
            raise ValueError("max_maps y max_display_size deben ser positivos.")
        current = bool(self.current_activations)
        maps = self.current_grad_heatmaps if current else self.last_step_grad_heatmaps
        # Orden de forward, aunque el backward recorra las capas al revés.
        items = sorted(maps.items(), key=lambda item: self.layer_to_index[item[0]])[:max_maps]
        if not items:
            print("No hay heatmaps de gradientes disponibles.")
            return
        rows = math.ceil(len(items) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 5), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (name, info) in zip(axes.ravel(), items):
            hm = info["heatmap"].detach().cpu().float()
            if not hm.numel():
                ax.set_title(f"{name}: sin valores")
                continue
            if vector_to_square and hm.shape[0] == 1 and hm.shape[1] > 1:
                side = math.ceil(math.sqrt(hm.numel()))
                hm = torch.nn.functional.pad(hm.flatten(), (0, side**2 - hm.numel()),
                                              value=float("nan")).reshape(side, side)
            h, w = hm.shape
            if h > max_display_size or w > max_display_size:
                # Promediar regiones para conservar la magnitud media, no elegir píxeles aislados.
                size = (min(h, max_display_size), min(w, max_display_size))
                valid = torch.isfinite(hm)
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    torch.where(valid, hm, 0)[None, None], size)[0, 0]
                weights = torch.nn.functional.adaptive_avg_pool2d(
                    valid.float()[None, None], size)[0, 0]
                hm = torch.where(weights > 0, pooled / weights, float("nan"))
            values = hm.numpy()
            h, w = values.shape
            ax.axis("on")
            im = ax.imshow(np.ma.masked_invalid(values), cmap="inferno",
                           interpolation="nearest", aspect="equal" if square_cells else "auto")
            ax.set_xticks(np.arange(w) if w <= 32 else [])
            ax.set_yticks(np.arange(h) if h <= 32 else [])
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which="minor", color="white", alpha=0.35, linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            finite = values[np.isfinite(values)]
            threshold = finite.max() / 2 if finite.size else 0
            if show_values and h * w <= max_values_to_show:
                for i in range(h):
                    for j in range(w):
                        if np.isfinite(values[i, j]):
                            ax.text(j, i, f"{values[i, j]:.2g}", ha="center", va="center",
                                    fontsize=6, color="black" if values[i, j] > threshold else "white")
            ax.set_title(f"{name}\ngrad={info['grad_shape']} | visual={tuple(hm.shape)}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        step = self.step_id if current else self.step_id - 1
        self._finish_plot(fig, f"gradient_heatmaps_grid_step_{step}.png", show, save)

    def _plot_history(self, source, metric, title, ylabel, filename, show, save):
        df = self.activations_df() if source == "activation" else self.gradients_df()
        if df.empty:
            print("No hay historial registrado.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for layer, group in df.groupby("layer", sort=False):
            ax.plot(group["step"], group[metric], label=layer)
        ax.set(title=title, xlabel="Step", ylabel=ylabel)
        ax.legend(fontsize=8, ncol=2)
        self._finish_plot(fig, filename, show, save)

    def plot_activation_history(self, show=True, save=True):
        self._plot_history("activation", "std", "Historial de std de activaciones",
                           "Std", "activation_std_history.png", show, save)

    def plot_gradient_history(self, show=True, save=True):
        self._plot_history("gradient", "abs_mean", "Historial de magnitud media de gradientes",
                           "Grad abs mean", "gradient_abs_mean_history.png", show, save)

    def detect_suspicious_layers(self, last_only=True, print_warnings=True,
                                 save=True, warn_missing_grads=True):
        act_df, grad_df = self.activations_df(), self.gradients_df()
        if last_only:
            act_df, grad_df = self._last_step_df(act_df), self._last_step_df(grad_df)
        issues = []

        def add(row, source, metric, severity, reason, value=None):
            value = row[metric] if value is None else value
            formatted = f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            issues.append(dict(step=int(row["step"]), severity=severity, source=source,
                               layer=row["layer"], metric=metric, value=value,
                               value_str=formatted, reason=reason))

        t = self.thresholds
        for source, df in (("activation", act_df), ("gradient", grad_df)):
            for _, row in df.iterrows():
                for metric in ("has_nan", "has_inf"):
                    if row[metric]:
                        add(row, source, metric, "CRITICAL", "Contiene valores no finitos.")
                for metric, threshold, reason in (
                    ("zero_ratio", f"{source}_zero_ratio", "Demasiados ceros."),
                    ("near_zero_ratio", f"{source}_near_zero_ratio", "Demasiados valores casi cero."),
                    ("abs_max", f"{source}_exploding_abs_max", "Magnitud máxima elevada."),
                    ("std", f"{source}_exploding_std", "Dispersión elevada."),
                ):
                    if np.isfinite(row[metric]) and row[metric] >= t[threshold]:
                        add(row, source, metric, "HIGH" if metric in ("abs_max", "std") else "MEDIUM", reason)
                if source == "activation" and row["std"] <= t["activation_dead_std"]:
                    dead = row["abs_mean"] <= t["activation_dead_abs_mean"]
                    add(row, source, "std", "HIGH" if dead else "MEDIUM",
                        "Posible capa muerta: std y magnitud casi cero." if dead else "Activación casi constante.")
                if source == "gradient" and row["abs_mean"] <= t["gradient_vanishing_abs_mean"]:
                    add(row, source, "abs_mean", "MEDIUM", "Gradiente de salida posiblemente desvanecido.")
        if warn_missing_grads and not act_df.empty:
            pairs = set(zip(grad_df["step"], grad_df["layer"])) if not grad_df.empty else set()
            for _, row in act_df.iterrows():
                if row["requires_grad"] and (row["step"], row["layer"]) not in pairs:
                    add(row, "gradient", "missing", "LOW",
                        "No se registró gradiente: revisa backward() y la conexión con la pérdida.", "missing")
        df = pd.DataFrame(issues, columns=self.ISSUE_COLUMNS)
        if not df.empty:
            df = df.assign(_severity=df["severity"].map(
                {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}))
            df = df.sort_values(["_severity", "step", "source", "layer"]).drop(columns="_severity")
        # Reemplazar el diagnóstico de los pasos consultados, sin duplicarlo.
        steps = {self.step_id - 1} if last_only else set(range(self.step_id))
        self.suspicious_history = [r for r in self.suspicious_history if r["step"] not in steps]
        self.suspicious_history.extend(df.to_dict("records"))
        if print_warnings:
            if df.empty:
                print("No hay datos registrados para este paso." if act_df.empty and grad_df.empty
                      else "No se detectaron anomalías con los umbrales actuales.")
            else:
                print(df[["severity", "source", "layer", "metric", "value_str", "reason"]].to_string(index=False))
        if save and not df.empty:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.save_dir / f"suspicious_layers_step_{int(df['step'].max())}.csv", index=False)
        return df

    def plot_suspicious_table(self, issues_df=None, max_rows=25, show=True, save=True):
        if max_rows < 1:
            raise ValueError("max_rows debe ser positivo.")
        if issues_df is None:
            issues_df = self.detect_suspicious_layers(print_warnings=False, save=False)
        if issues_df.empty:
            print("No hay capas sospechosas para graficar.")
            return
        df = issues_df[["severity", "source", "layer", "metric", "value_str", "reason"]].head(max_rows)
        fig, ax = plt.subplots(figsize=(14, max(3, 0.45 * len(df) + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        ax.set_title("Capas sospechosas", pad=20)
        self._finish_plot(fig, f"suspicious_layers_table_step_{int(issues_df['step'].max())}.png", show, save)

    @staticmethod
    def _normalize_images_for_display(images):
        images = images.clone()
        for i, image in enumerate(images):
            lo, hi = image.min(), image.max()
            images[i] = (image - lo) / (hi - lo) if hi - lo > 1e-8 else torch.zeros_like(image)
        return images.clamp(0, 1)

    def show_input_rgb(self, channels=(0, 1, 2), max_images=4,
                       normalize=True, show=True, save=True):
        if max_images < 1:
            raise ValueError("max_images debe ser positivo.")
        x = self.last_input_batch
        if x is None or x.ndim != 4 or any(s == 0 for s in x.shape):
            print("No hay un batch de imágenes válido (B, C, H, W) capturado.")
            return
        images = x[:max_images].detach().cpu().float()
        c = images.shape[1]
        if c >= 3:
            if len(channels) != 3 or any(not isinstance(i, int) or not 0 <= i < c for i in channels):
                raise ValueError("channels debe contener tres índices de canal válidos.")
            images = images[:, list(channels)]
        elif c == 2:
            images = images[:, [0, 1, 0]]
        else:
            images = images.repeat(1, 3, 1, 1)
        if normalize:
            images = self._normalize_images_for_display(images)
        cols = min(len(images), 4)
        rows = math.ceil(len(images) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, image in enumerate(images):
            axes.ravel()[i].imshow(image.permute(1, 2, 0).numpy())
            axes.ravel()[i].set_title(f"Input {i}")
        step = self.step_id if self.current_activations else self.step_id - 1
        self._finish_plot(fig, f"input_rgb_step_{step}.png", show, save)

    def save_tables(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name, df in (("activations", self.activations_df()),
                         ("gradients", self.gradients_df()),
                         ("suspicious_history", self.suspicious_df())):
            if not df.empty:
                df.to_csv(self.save_dir / f"{name}.csv", index=False)

    def report(self, show=True, save=True, include_ranges=True, include_history=False,
               include_heatmaps=True, include_input=True, include_histograms=True,
               include_teaching_plots=True, include_suspicious=True, plot_suspicious=True):
        """Genera el reporte después de step(); devuelve las alertas en un DataFrame."""
        self.plot_activation_stats(show=show, save=save)
        self.plot_gradient_stats(show=show, save=save)
        if include_teaching_plots:
            self.plot_activation_std_flow(show=show, save=save)
            self.plot_activation_alive_ratio(show=show, save=save)
            self.plot_gradient_abs_mean_flow(show=show, save=save)
        if include_histograms:
            self.plot_activation_histograms(show=show, save=save)
        if include_ranges:
            self.plot_activation_min_max(show=show, save=save)
            self.plot_gradient_min_max(show=show, save=save)
        if include_heatmaps:
            self.plot_gradient_heatmaps(show=show, save=save)
        if include_input:
            self.show_input_rgb(show=show, save=save)
        if include_history:
            self.plot_activation_history(show=show, save=save)
            self.plot_gradient_history(show=show, save=save)
        suspicious = None
        if include_suspicious:
            suspicious = self.detect_suspicious_layers(save=save)
            if plot_suspicious and not suspicious.empty:
                self.plot_suspicious_table(suspicious, show=show, save=save)
        if save:
            self.save_tables()
        return suspicious

    def close(self):
        for handle in self.handles + self._tensor_handles:
            handle.remove()
        self.handles.clear()
        self._tensor_handles.clear()


def compare_monitors(monitors, labels=None, metric="activation_std", show=True,
                     save=True, save_dir="monitor_comparison"):
    """Compara el último paso de varios monitores; admite un dict nombre: monitor."""
    metrics = {
        "activation_std": ("std", "Comparación: std de activaciones", "Std"),
        "activation_alive": ("alive_ratio", "Comparación: valores no cercanos a cero", "Porcentaje"),
        "gradient_abs_mean": ("abs_mean", "Comparación: magnitud media de gradientes", "Grad abs mean"),
    }
    if metric not in metrics:
        raise ValueError(f"metric debe ser uno de {tuple(metrics)}.")
    if isinstance(monitors, dict):
        labels, monitors = list(monitors.keys()), list(monitors.values())
    else:
        monitors = list(monitors)
        labels = list(labels) if labels is not None else [f"run_{i}" for i in range(len(monitors))]
    if len(labels) != len(monitors) or len(set(labels)) != len(labels):
        raise ValueError("Debe haber una etiqueta única por monitor.")
    column, title, ylabel = metrics[metric]
    rows = []
    for label, monitor in zip(labels, monitors):
        df = monitor.gradients_df() if metric == "gradient_abs_mean" else monitor.activations_df()
        df = monitor._sort_df_by_forward_order(monitor._last_step_df(df))
        for _, row in df.iterrows():
            rows.append(dict(run=label, layer=row["layer"], layer_index=row["layer_index"],
                             value=row[column] * (100 if metric == "activation_alive" else 1)))
    comp_df = pd.DataFrame(rows, columns=["run", "layer", "layer_index", "value"])
    if comp_df.empty:
        print("No hay datos para comparar.")
        return comp_df
    layers = comp_df.sort_values("layer_index", kind="stable")["layer"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.45), 5))
    for label in labels:
        values = comp_df[comp_df["run"] == label].set_index("layer")["value"].reindex(layers).to_numpy()
        if metric == "gradient_abs_mean":
            values = np.clip(values, 1e-20, None)
        ax.plot(np.arange(len(layers)), values, marker="o", label=label)
    if metric == "gradient_abs_mean":
        ax.set_yscale("log")
    if metric == "activation_alive":
        ax.set_ylim(-2, 102)
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=75, ha="right")
    ax.set(title=title, ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"compare_{metric}.png", dpi=150)
        comp_df.to_csv(save_dir / f"compare_{metric}.csv", index=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return comp_df
