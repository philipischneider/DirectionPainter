from __future__ import annotations
import numpy as np
from .scene import DirectionLayer


def export_layer(layer: DirectionLayer, path: str) -> None:
    """
    Export a direction layer as a PNG file.

    Channel mapping (standard 3D flow map convention):
        R = cos(θ) remapped [−1, 1] → [0, 65535]  (16-bit if pypng available)
        G = sin(θ) remapped [−1, 1] → [0, 65535]
        B = intensity falloff [0, 1] → [0, 65535]

    Falls back to 8-bit if pypng is not installed.
    """
    tex = layer.texture  # (H, W, 3) float32
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

    # R and G are in [-1, 1]; remap to [0, 65535].
    rg_remap = ((tex[..., :2] + 1.0) * 0.5 * 65535.0).clip(0, 65535).astype(np.uint16)
    # B is already in [0, 1]; remap to [0, 65535].
    b_remap  = (tex[..., 2] * 65535.0).clip(0, 65535).astype(np.uint16)
    rows = []
    for y in range(H):
        row = np.zeros(W * 3, dtype=np.uint16)
        row[0::3] = rg_remap[y, :, 0]   # R = cos(θ)
        row[1::3] = rg_remap[y, :, 1]   # G = sin(θ)
        row[2::3] = b_remap[y, :]        # B = intensity
        rows.append(row.tolist())

    with open(path, "wb") as f:
        writer = png.Writer(width=W, height=H, bitdepth=16, greyscale=False)
        writer.write(f, rows)


def _export_8bit(tex: np.ndarray, H: int, W: int, path: str) -> None:
    from PIL import Image

    # R and G are in [-1, 1]; remap to [0, 255].
    rg_remap = ((tex[..., :2] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    # B is already in [0, 1]; remap to [0, 255].
    b_remap  = (tex[..., 2] * 255.0).clip(0, 255).astype(np.uint8)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, :, 0] = rg_remap[:, :, 0]   # R = cos(θ)
    out[:, :, 1] = rg_remap[:, :, 1]   # G = sin(θ)
    out[:, :, 2] = b_remap             # B = intensity
    Image.fromarray(out, mode="RGB").save(path, format="PNG")
