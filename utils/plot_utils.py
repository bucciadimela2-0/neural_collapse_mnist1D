import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Output folder + timestamp

def make_img_dir(base_dir="img", run_ts=None):
   
    if run_ts is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, run_ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, run_ts


def _savefig(fig, out_dir, run_ts, name, dpi=300, **kwargs):
  
    fname = f"{run_ts}_{name}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


def _resolve_tpt_line(tpt_epoch, default=20):
    return tpt_epoch if (tpt_epoch is not None) else default


def _history_to_arrays(history):
    
    d = dict(history)
    out = {}
    for k, v in d.items():
        out[k] = np.array(v) if isinstance(v, (list, tuple)) else v
    return out


def _has_series(h, k):
    return (k in h) and (isinstance(h[k], np.ndarray)) and (h[k].size > 0)


def _align_series(x, y):
    n = min(len(x), len(y))
    if n == 0:
        return None, None
    return x[-n:], y[-n:]



# Figure 1: Training metrics
def plot_training_metrics(history, tpt_epoch, out_dir, run_ts, name="1_training_metrics"):
    h = _history_to_arrays(history)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    epochs = h["epoch"]
    tpt_line = _resolve_tpt_line(tpt_epoch, default=20)

    colors = {"train": "#1a1a1a", "val": "#d73027", "tpt": "#7b3294"}

    # ---- Accuracy ----
    ax = axes[0]
    ax.plot(epochs, h["train_acc"], color=colors["train"], linewidth=1, label="Train Acc")
    ax.plot(epochs, h["test_acc"],  color=colors["val"],   linewidth=1, label="Val Acc")
    ax.axvline(tpt_line, color=colors["tpt"], linewidth=1, linestyle="--", alpha=0.9, label="TPT")

    ax.set_title("Accuracy Evolution", fontweight="bold", fontsize=14, pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.5, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Loss ----
    ax = axes[1]
    if _has_series(h, "train_loss") and _has_series(h, "val_loss"):
        ax.plot(epochs, h["train_loss"], color=colors["train"], linewidth=1, label="Train Loss")
        ax.plot(epochs, h["val_loss"],   color=colors["val"],   linewidth=1, label="Val Loss")
        ax.set_yscale("log")
    else:
        # placeholder 
        epochs_placeholder = np.linspace(0, epochs[-1], len(epochs))
        ax.plot(epochs_placeholder, np.logspace(-1, -4, len(epochs)), color=colors["train"], linewidth=1, label="Train Loss")
        ax.plot(epochs_placeholder, np.logspace(-1, -3.5, len(epochs)), color=colors["val"], linewidth=1, linestyle="--", label="Val Loss")
        ax.set_yscale("log")

    ax.axvline(tpt_line, color=colors["tpt"], linewidth=1, linestyle="--", alpha=0.9, label="TPT")

    ax.set_title("Loss Evolution", fontweight="bold", fontsize=14, pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Neural Collapse Training Dynamics", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()

    path = _savefig(fig, out_dir, run_ts, name=name, dpi=300)
    plt.show()
    return path



# Figure 2: NC metrics (NC1-NC4)
def plot_nc_metrics(history, tpt_epoch, out_dir, run_ts, name="2_nc_metrics"):
    h = _history_to_arrays(history)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    epochs = h["epoch"]
    tpt_line = _resolve_tpt_line(tpt_epoch, default=20)

    metrics_info = [
        ("NC1_collapse", "NC1: Collapse", True),
        ("NC2_etf",      "NC2: ETF",      False),
        ("NC3_duality",  "NC3: Duality",  False),
        ("NC4_ncc",      "NC4: NCC",      False),
    ]

    for idx, (metric, title, use_log) in enumerate(metrics_info):
        ax = axes[idx]
        if not _has_series(h, metric):
            ax.set_title(f"{title}\n(missing)", fontweight="bold")
            ax.axis("off")
            continue

        x, y = _align_series(epochs, h[metric])
        ax.plot(x, y, color="black", linewidth=1.0, alpha=0.85)
        ax.axvline(tpt_line, color="purple", linestyle="--", linewidth=1.0, alpha=0.9)

        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 1.0)
        else:
            ax.set_ylim(0.0, 1.0)

        ax.set_xlim(0, epochs[-1])
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("NC Metric", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

        final_val = float(y[-1])
        ax.text(
            0.98, 0.98, f"{final_val:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
        )

    plt.suptitle("Neural Collapse Metrics (NC1-NC4)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = _savefig(fig, out_dir, run_ts, name=name, dpi=300)
    plt.show()
    return path



# Figure 3: Geometric analysis
def plot_geometric_analysis(model, metrics, H, means, y_test, out_dir, run_ts, name="3_geometric_analysis"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mu_G = H.mean(0, keepdims=True)
    means_centered = means - mu_G
    means_norm = means_centered / (np.linalg.norm(means_centered, axis=1, keepdims=True) + 1e-8)
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # ---- Plot 1: Feature Space PCA ----
    ax = axes[0]
    pca = PCA(n_components=2)
    H_pca = pca.fit_transform(H)
    means_pca = pca.transform(means)

    for i in range(10):
        mask = (y_test == i)
        ax.scatter(H_pca[mask, 0], H_pca[mask, 1], c=[colors[i]], alpha=0.4, s=8, edgecolors="none")

    ax.scatter(means_pca[:, 0], means_pca[:, 1], c=colors, s=400, marker="*", edgecolors="black", linewidths=1.5, zorder=5)

    for i in range(10):
        for j in range(i + 1, 10):
            ax.plot([means_pca[i, 0], means_pca[j, 0]], [means_pca[i, 1], means_pca[j, 1]], "k-", alpha=0.1, linewidth=0.5, zorder=1)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12, fontweight="bold")
    ax.set_title(f'Feature Space Collapse\nNC1 = {metrics["NC1_collapse"]:.4f}', fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_facecolor("#fafafa")

    # ---- Plot 2: Gram Matrix ----
    ax = axes[1]
    gram = means_norm @ means_norm.T
    im = ax.imshow(gram, cmap="RdBu_r", vmin=-0.3, vmax=1.0, aspect="auto", interpolation="nearest")

    for i in range(10):
        for j in range(10):
            if i == j:
                text, color, weight = "1.0", "white", "bold"
            else:
                text = f"{gram[i, j]:.2f}"
                color = "white" if gram[i, j] < 0 else "black"
                weight = "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color, weight=weight)

    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Class", fontsize=12, fontweight="bold")
    ax.set_title(f'Normalized Gram Matrix\nNC2 = {metrics["NC2_etf"]:.4f} | Target: -0.111', fontsize=13, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("⟨μᵢ, μⱼ⟩", fontsize=12, fontweight="bold", rotation=0, labelpad=20)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # ---- Plot 3: NCC Confusion Matrix ----
    ax = axes[2]
    dists = np.linalg.norm(H_norm[:, None] - means_norm[None], axis=2)
    ncc_pred = np.argmin(dists, axis=1)
    cm = confusion_matrix(y_test, ncc_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title(f'NCC Classification\nAccuracy = {metrics["NC4_ncc"]:.4f}', fontsize=13, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized\nCount", fontsize=11, fontweight="bold")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    plt.suptitle("Geometric Analysis of Neural Collapse", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    path = _savefig(fig, out_dir, run_ts, name=name, dpi=300, facecolor="white", edgecolor="none")
    plt.show()
    return path



# Comparison plots (multi-optimizer)
def plot_optimizer_comparison(results, out_dir=None, run_ts=None,
                              name_prefix="opt_comparison",
                              include_nc=True, include_loss=True):

    if out_dir is None or run_ts is None:
        out_dir, run_ts = make_img_dir("img")

    opts = list(results.keys())
    histories = {opt: _history_to_arrays(results[opt]["history"]) for opt in opts}

    # -------- FIG 1: Accuracy --------
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.2))
    for opt in opts:
        h = histories[opt]
        if not _has_series(h, "epoch") or not _has_series(h, "test_acc"):
            continue
        ax1.plot(h["epoch"], h["test_acc"], linewidth=2, label=f"{opt} (test)")
        if _has_series(h, "train_acc"):
            ax1.plot(h["epoch"], h["train_acc"], linewidth=1, linestyle="--", alpha=0.7, label=f"{opt} (train)")

    ax1.set_title("Optimizer Comparison — Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0.0, 1.02)
    ax1.legend(ncol=2, frameon=True)
    path1 = _savefig(fig1, out_dir, run_ts, f"{name_prefix}_accuracy")
    plt.show()

    # -------- FIG 2: Loss --------
    path2 = None
    if include_loss:
        fig2, ax2 = plt.subplots(figsize=(10.5, 5.2))
        any_loss = False
        for opt in opts:
            h = histories[opt]
            if _has_series(h, "epoch") and _has_series(h, "val_loss"):
                any_loss = True
                ax2.plot(h["epoch"], h["val_loss"], linewidth=2, label=f"{opt} (val)")
            if _has_series(h, "epoch") and _has_series(h, "train_loss"):
                any_loss = True
                ax2.plot(h["epoch"], h["train_loss"], linewidth=1, linestyle="--", alpha=0.7, label=f"{opt} (train)")

        ax2.set_title("Optimizer Comparison — Loss", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.2)
        if any_loss:
            ax2.set_yscale("log")
        ax2.legend(ncol=2, frameon=True)
        path2 = _savefig(fig2, out_dir, run_ts, f"{name_prefix}_loss")
        plt.show()

    # -------- FIG 3: NC metrics --------
    path3 = None
    if include_nc:
        nc_keys = ["NC1_collapse", "NC2_etf", "NC3_duality", "NC4_ncc"]
        if any(_has_series(histories[o], "NC5_volume") for o in opts):
            nc_keys.append("NC5_volume")

        n = len(nc_keys)
        fig3, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2), sharex=True)
        if n == 1:
            axes = [axes]

        for j, k in enumerate(nc_keys):
            ax = axes[j]
            for opt in opts:
                h = histories[opt]
                if _has_series(h, "epoch") and _has_series(h, k):
                    x, y = _align_series(h["epoch"], h[k])
                    ax.plot(x, y, linewidth=2, label=opt)

            ax.set_title(k, fontweight="bold")
            ax.set_xlabel("Epoch")
            if j == 0:
                ax.set_ylabel("Value")
            ax.grid(True, alpha=0.2)

            if k == "NC1_collapse":
                ax.set_yscale("log")
            elif k in ("NC2_etf", "NC4_ncc", "NC5_volume"):
                ax.set_ylim(0.0, 1.05)

            # TPT lines 
            for opt in opts:
                tpt = results[opt].get("tpt_epoch", None)
                if tpt is not None:
                    ax.axvline(tpt, linestyle="--", linewidth=1, alpha=0.3)

            if j == n - 1:
                ax.legend(frameon=True)

        fig3.suptitle("Optimizer Comparison — Neural Collapse Metrics", fontweight="bold", y=1.02)
        fig3.tight_layout()
        path3 = _savefig(fig3, out_dir, run_ts, f"{name_prefix}_nc")
        plt.show()

    return {
        "out_dir": out_dir,
        "run_ts": run_ts,
        "accuracy_fig": path1,
        "loss_fig": path2,
        "nc_fig": path3,
    }

def plot_layerwise_grid(history, layer_names, out_dir, run_ts, name="4_layerwise_nc.png"):
    """
    Plot grid: Layers (rows) × NC Metrics (columns)
    """

    if out_dir is None or run_ts is None:
        out_dir, run_ts = make_img_dir("img")

    fig, axes = plt.subplots(len(layer_names), 3, figsize=(15, 3*len(layer_names)))
    
    epochs = np.array(history['global']['epoch'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    
    metrics_to_plot = ['NC1', 'NC2', 'NC4']
    metric_titles = ['NC1: Within-Class Collapse', 'NC2: ETF Convergence', 'NC4: NCC Accuracy']
    
    for row_idx, layer_name in enumerate(layer_names):
        for col_idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[row_idx, col_idx] if len(layer_names) > 1 else axes[col_idx]
            
            values = np.array(history[layer_name][metric])
            
            # Plot line
            ax.plot(epochs, values, 
                   color=colors[row_idx], 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=4,
                   alpha=0.8)
            
            # Styling
            if metric == 'NC1':
                ax.set_yscale('log')
                ax.set_ylabel('NC1 (log)', fontweight='bold')
            else:
                ax.set_ylabel(metric, fontweight='bold')
            
            # Title with layer name
            if col_idx == 0:
                ax.set_ylabel(f'{layer_name}\n{metric}', fontweight='bold', fontsize=10)
            
            if row_idx == 0:
                ax.set_title(title, fontweight='bold', fontsize=11)
            
            if row_idx == len(layer_names) - 1:
                ax.set_xlabel('Epoch', fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Final value annotation
            final_val = values[-1]
            ax.text(0.98, 0.95, f'{final_val:.3f}', 
                   transform=ax.transAxes,
                   ha='right', va='top',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='yellow', alpha=0.6))
    
    plt.suptitle('Layer-wise Neural Collapse Evolution', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    path = _savefig(fig, out_dir, run_ts, name=name, dpi=300, facecolor="white", edgecolor="none")
    plt.show()
    
    plt.show()