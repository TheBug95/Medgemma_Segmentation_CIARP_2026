# =============================================================================
# scripts/visualize.py
# =============================================================================
# Feedback visual: Grad-CAM vs máscara GT
#
# VISUALIZACIÓN 2x2 POR IMAGEN:
#   (a) Imagen original + título con predicción
#   (b) Heatmap de Grad-CAM superpuesto
#   (c) Máscara GT binarizada
#   (d) Overlay: GT en azul, Grad-CAM contour en rojo
#
# USO:
#   from scripts.visualize import plot_gradcam_vs_gt
#   plot_gradcam_vs_gt(image, gradcam, gt_mask, image_id, save_path)
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gradcam_vs_gt(
    image: np.ndarray,
    gradcam: np.ndarray,
    gt_mask: np.ndarray,
    image_id: str,
    save_path: str = None,
    figure_size: tuple[float, float] = (12, 10),
    show: bool = False,
    percentile: int = 95,
) -> None:
    """
    Genera visualización 2x2 comparando Grad-CAM vs máscara GT.

    Args:
        image: ndarray (H, W, 3) en [0, 1], imagen original
        gradcam: ndarray (H, W) en [0, 1], heatmap de Grad-CAM
        gt_mask: ndarray (H, W), máscara GT (0=bg, 1=OD)
        image_id: identificador de la imagen
        save_path: ruta donde guardar la figura (None = no guardar)
        figure_size: tamaño de figura en pulgadas
        show: si True, llama plt.show()
        percentile: percentil usado para binarizar el contour en el panel (d)
    """
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    fig.suptitle(f"Grad-CAM vs GT - {image_id} (p{percentile})", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("(a) Imagen original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    axes[0, 1].imshow(gradcam, cmap="jet", alpha=0.6)
    axes[0, 1].set_title("(b) Grad-CAM heatmap")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gt_mask, cmap="gray")
    axes[1, 0].set_title("(c) Máscara GT (binaria)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(image)

    gt_binary = (gt_mask > 0).astype(np.uint8)
    gt_dilated = np.zeros_like(gt_binary)
    if gt_binary.sum() > 0:
        from scipy.ndimage import binary_dilation

        structure = np.ones((15, 15))
        gt_dilated = binary_dilation(gt_binary, structure).astype(np.uint8)

    rows, cols = np.where(gt_dilated > 0)
    if len(rows) > 0:
        axes[1, 1].scatter(cols, rows, c="blue", s=3, alpha=0.3, marker="s", label="GT (dilated)")

    # Usar el percentil pasado como parámetro para el contour
    thresh = np.percentile(gradcam, percentile)
    contour_mask = (gradcam >= thresh).astype(np.uint8)
    import scipy.ndimage as ndimage
    contour_outline = contour_mask - ndimage.binary_erosion(contour_mask)
    rows_c, cols_c = np.where(contour_outline > 0)
    if len(rows_c) > 0:
        axes[1, 1].scatter(cols_c, rows_c, c="red", s=3, marker="o", label=f"Grad-CAM p{percentile}")

    axes[1, 1].set_title(f"(d) Overlay: GT (blue) vs Grad-CAM contour p{percentile}")
    axes[1, 1].axis("off")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


