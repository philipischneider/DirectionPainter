from __future__ import annotations
import numpy as np
from .scene import DirectionLayer


def export_layer(layer: DirectionLayer, path: str) -> None:
    """
    Export a direction layer as a PNG file.

    Channel mapping (same convention as a normal map):
        R = cos(θ) remapped [−1, 1] → [0, 65535]  (16-bit if pypng available)
        G = sin(θ) remapped [−1, 1] → [0, 65535]
        B = 0

    Falls back to 8-bit if pypng is not installed.
    """
    tex = layer.texture  # (H, W, 2) float32, values in [−1, 1]
    H, W = tex.shape[:2]

    # Try 16-bit PNG via pypng
    try:
        import png
        _export_16bit(tex, H, W, path)
        return
    except ImportError:
        pass

    # Fallback: 8-bit PNG via Pillow
    _export_8bit(tex, H, W, path)


# ─────────────────────────────────────────────────────────────────────────────

def _export_16bit(tex: np.ndarray, H: int, W: int, path: str) -> None:
    import png

    remap = ((tex + 1.0) * 0.5 * 65535.0).clip(0, 65535).astype(np.uint16)
    rows = []
    for y in range(H):
        row = np.zeros(W * 3, dtype=np.uint16)
        row[0::3] = remap[y, :, 0]   # R = cos(θ)
        row[1::3] = remap[y, :, 1]   # G = sin(θ)
        # B stays 0
        rows.append(row.tolist())

    with open(path, "wb") as f:
        writer = png.Writer(width=W, height=H, bitdepth=16, greyscale=False)
        writer.write(f, rows)


def _export_8bit(tex: np.ndarray, H: int, W: int, path: str) -> None:
    from PIL import Image

    remap = ((tex + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, :, 0] = remap[:, :, 0]   # R = cos(θ)
    out[:, :, 1] = remap[:, :, 1]   # G = sin(θ)
    Image.fromarray(out, mode="RGB").save(path, format="PNG")
