import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.segmentation import CLUSTER_COLORS, CLUSTER_NAMES


def overlay_segmentation(
    img_bgr: np.ndarray,
    color_map_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend RGB color_map over original BGR image."""
    color_map_bgr = cv2.cvtColor(color_map_rgb, cv2.COLOR_RGB2BGR)
    cm_h, cm_w    = color_map_bgr.shape[:2]
    overlay       = img_bgr.copy()
    overlay[:cm_h, :cm_w] = cv2.addWeighted(
        img_bgr[:cm_h, :cm_w], 1 - alpha,
        color_map_bgr,          alpha,
        0,
    )
    return overlay


def save_segmentation_figure(
    img_bgr: np.ndarray,
    color_map_raw: np.ndarray,
    color_map_smooth: np.ndarray,
    overlay: np.ndarray,
    labels: np.ndarray,
    labels_smooth: np.ndarray,
    n_clusters: int,
    output_path: str,
) -> None:
    """Save 4-panel figure: Original | Raw | Smoothed | Overlay."""

    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    fig.patch.set_facecolor("#1e1e1e")

    titles = [
        "Original Image",
        "KMeans (raw)",
        "KMeans + Smoothing (5x5 median)",
        "Final Overlay",
    ]
    images = [
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        color_map_raw,
        color_map_smooth,
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="white", pad=10)
        ax.axis("off")

    # Legend with smoothed label counts
    legend_patches = []
    for i in range(n_clusters):
        count = int(np.sum(labels_smooth == i))
        pct   = count / len(labels_smooth) * 100
        color_norm = [c / 255 for c in CLUSTER_COLORS[i]]
        legend_patches.append(
            mpatches.Patch(
                color=color_norm,
                label=f"{CLUSTER_NAMES[i]}  —  {count} patches ({pct:.1f}%)",
            )
        )

    axes[2].legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=n_clusters,
        fontsize=10,
        framealpha=0.85,
        facecolor="#2a2a2a",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Step 5] Saved → {output_path}")