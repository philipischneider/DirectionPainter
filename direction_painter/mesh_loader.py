"""
mesh_loader.py — Load GLTF/GLB into a Mesh.

trimesh handles geometry (vertices, faces, normals).
TEXCOORD_0 and TANGENT are read directly from the binary GLTF chunk because
trimesh 4.x drops those attributes when no texture material is present.
"""
from __future__ import annotations
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

from .scene import Mesh


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_gltf(path: str) -> Mesh:
    """Load a GLTF/GLB file and return a Mesh with all required vertex attributes."""
    p = Path(path)
    raw = p.read_bytes()

    # Parse the binary GLTF chunk to extract all primitive attributes reliably
    gltf_attrs = _parse_gltf_attributes(raw, p.suffix.lower())

    # Use trimesh only for processed geometry (merged vertices, normals)
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.geometry.values())
        if not geos:
            raise ValueError("GLTF scene contains no geometry")
        tm = trimesh.util.concatenate(geos)
    elif isinstance(loaded, trimesh.Trimesh):
        tm = loaded
    else:
        raise ValueError(f"Unsupported trimesh type: {type(loaded)}")

    vertices = np.array(tm.vertices,       dtype=np.float32)
    indices  = np.array(tm.faces,          dtype=np.uint32)
    normals  = np.array(tm.vertex_normals, dtype=np.float32)

    # UV ──────────────────────────────────────────────────────────────────────
    uvs = gltf_attrs.get("TEXCOORD_0")
    if uvs is None:
        uvs = _extract_uvs_trimesh(tm)   # last-resort: trimesh fallback
    uvs = uvs.astype(np.float32)

    # Validate UV vertex count matches mesh vertex count
    if len(uvs) != len(vertices):
        raise ValueError(
            f"UV count ({len(uvs)}) does not match vertex count ({len(vertices)}). "
            "Make sure the mesh has no un-applied modifiers or multiple materials "
            "that cause re-indexing when exported."
        )

    # Tangents ────────────────────────────────────────────────────────────────
    tangents = gltf_attrs.get("TANGENT")
    if tangents is None or len(tangents) != len(vertices):
        tangents = _compute_tangents(vertices, normals, uvs, indices)
    else:
        tangents = tangents.astype(np.float32)

    # Bitangents: cross(normal, tangent.xyz) * tangent.w
    bitangents = np.cross(normals, tangents[:, :3]) * tangents[:, 3:4]
    mag = np.linalg.norm(bitangents, axis=1, keepdims=True)
    bitangents /= np.maximum(mag, 1e-8)
    bitangents = bitangents.astype(np.float32)

    # Per-face centroid data for tick overlay
    face_uv_centroids  = uvs[indices].mean(axis=1).astype(np.float32)
    face_pos_centroids = vertices[indices].mean(axis=1).astype(np.float32)

    face_tan = tangents[indices, :3].mean(axis=1).astype(np.float32)
    face_tan /= np.maximum(np.linalg.norm(face_tan, axis=1, keepdims=True), 1e-8)

    face_bitan = bitangents[indices].mean(axis=1).astype(np.float32)
    face_bitan /= np.maximum(np.linalg.norm(face_bitan, axis=1, keepdims=True), 1e-8)

    uv_aspect, uv_tan_len, face_uv_aspect, face_uv_tan_len = _compute_uv_aspect(vertices, uvs, indices)

    return Mesh(
        vertices=vertices,
        normals=normals,
        tangents=tangents,
        uvs=uvs,
        indices=indices,
        bitangents=bitangents,
        uv_aspect=uv_aspect,
        uv_tan_len=uv_tan_len,
        face_uv_aspect=face_uv_aspect,
        face_uv_tan_len=face_uv_tan_len,
        face_uv_centroids=face_uv_centroids,
        face_pos_centroids=face_pos_centroids,
        face_tangents=face_tan,
        face_bitangents=face_bitan,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Direct GLTF binary parser
# ─────────────────────────────────────────────────────────────────────────────

# GLTF accessor component types → (numpy dtype, bytes per element)
_COMPONENT_TYPE = {
    5120: (np.int8,    1),
    5121: (np.uint8,   1),
    5122: (np.int16,   2),
    5123: (np.uint16,  2),
    5125: (np.uint32,  4),
    5126: (np.float32, 4),
}

_ACCESSOR_TYPE_COLS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4,   "MAT3": 9, "MAT4": 16,
}


def _parse_gltf_attributes(raw: bytes, ext: str) -> dict[str, np.ndarray]:
    """
    Parse a GLB or GLTF file and return a dict of {attributeName: np.ndarray}
    for all attributes found in the first mesh primitive.
    """
    if ext == ".glb":
        json_dict, bin_chunk = _split_glb(raw)
    else:
        json_dict = json.loads(raw.decode("utf-8"))
        bin_chunk = b""

    buffer_views = json_dict.get("bufferViews", [])
    accessors    = json_dict.get("accessors", [])
    meshes       = json_dict.get("meshes", [])

    if not meshes:
        return {}

    def read_accessor(idx: int) -> np.ndarray:
        acc     = accessors[idx]
        bv_idx  = acc.get("bufferView")
        if bv_idx is None:
            return np.array([], dtype=np.float32)

        bv           = buffer_views[bv_idx]
        bv_off       = bv.get("byteOffset", 0)
        comp_type_id = acc["componentType"]
        dtype, comp_bytes = _COMPONENT_TYPE[comp_type_id]
        ncols        = _ACCESSOR_TYPE_COLS[acc["type"]]
        count        = acc["count"]
        byte_stride  = bv.get("byteStride", comp_bytes * ncols)
        acc_off      = acc.get("byteOffset", 0)
        start        = bv_off + acc_off

        chunk = bin_chunk[start : start + count * byte_stride]

        if byte_stride == comp_bytes * ncols:
            arr = np.frombuffer(chunk, dtype=dtype).reshape(count, ncols)
        else:
            # Interleaved buffer
            arr = np.zeros((count, ncols), dtype=dtype)
            for i in range(count):
                off = i * byte_stride
                arr[i] = np.frombuffer(
                    chunk[off : off + comp_bytes * ncols], dtype=dtype
                )

        return arr.astype(np.float32) if dtype != np.float32 else arr.copy()

    # Use the first primitive of the first mesh
    prim  = meshes[0]["primitives"][0]
    attrs = prim.get("attributes", {})

    result: dict[str, np.ndarray] = {}
    for attr_name, accessor_idx in attrs.items():
        result[attr_name] = read_accessor(accessor_idx)

    # Flatten SCALAR columns
    for k in list(result.keys()):
        if result[k].ndim == 2 and result[k].shape[1] == 1:
            result[k] = result[k].reshape(-1)

    return result


def _split_glb(data: bytes) -> tuple[dict, bytes]:
    """Split a GLB binary into its JSON dict and binary buffer."""
    magic, version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:   # 'glTF' little-endian
        raise ValueError("Not a valid GLB file (bad magic bytes)")

    offset    = 12
    json_dict: dict = {}
    bin_chunk: bytes = b""

    while offset < len(data):
        if offset + 8 > len(data):
            break
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_len]
        offset    += chunk_len

        if chunk_type == 0x4E4F534A:    # JSON
            json_dict = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == 0x004E4942:  # BIN\0
            bin_chunk = chunk_data

    return json_dict, bin_chunk


# ─────────────────────────────────────────────────────────────────────────────
# Fallback UV extraction via trimesh
# ─────────────────────────────────────────────────────────────────────────────

def _extract_uvs_trimesh(tm: trimesh.Trimesh) -> np.ndarray:
    """Try every trimesh path to get UV data. Raises if none found."""
    for key in ("TEXCOORD_0", "texcoord_0", "_TEXCOORD_0"):
        if key in tm.vertex_attributes:
            return np.array(tm.vertex_attributes[key], dtype=np.float32)

    if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
        return np.array(tm.visual.uv, dtype=np.float32)

    raise ValueError(
        "Mesh has no UV channel. "
        "In Blender: File → Export → glTF 2.0 → Geometry → check 'UVs'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tangent computation fallback
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tangents(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex tangents from UV mapping (MikkTSpace approximation)."""
    N = len(vertices)
    tan_accum  = np.zeros((N, 3), dtype=np.float64)
    btan_accum = np.zeros((N, 3), dtype=np.float64)

    v0  = vertices[indices[:, 0]].astype(np.float64)
    v1  = vertices[indices[:, 1]].astype(np.float64)
    v2  = vertices[indices[:, 2]].astype(np.float64)
    uv0 = uvs[indices[:, 0]].astype(np.float64)
    uv1 = uvs[indices[:, 1]].astype(np.float64)
    uv2 = uvs[indices[:, 2]].astype(np.float64)

    e1  = v1 - v0
    e2  = v2 - v0
    du1 = (uv1 - uv0)[:, 0:1]
    dv1 = (uv1 - uv0)[:, 1:2]
    du2 = (uv2 - uv0)[:, 0:1]
    dv2 = (uv2 - uv0)[:, 1:2]

    denom = du1 * dv2 - du2 * dv1
    r     = np.where(np.abs(denom) > 1e-12, 1.0 / denom, 0.0)

    tan  = r * (dv2 * e1 - dv1 * e2)
    btan = r * (du1 * e2 - du2 * e1)

    for i in range(3):
        np.add.at(tan_accum,  indices[:, i], tan)
        np.add.at(btan_accum, indices[:, i], btan)

    n     = normals.astype(np.float64)
    t     = tan_accum - n * (np.sum(n * tan_accum, axis=1, keepdims=True))
    t_mag = np.linalg.norm(t, axis=1, keepdims=True)
    t     = t / np.maximum(t_mag, 1e-8)

    cross = np.cross(n, tan_accum)
    w     = np.where(np.sum(cross * btan_accum, axis=1) < 0, -1.0, 1.0)

    return np.hstack([t.astype(np.float32),
                      w.reshape(-1, 1).astype(np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# UV aspect ratio (|∂pos/∂u| / |∂pos/∂v|) per vertex
# ─────────────────────────────────────────────────────────────────────────────

def _compute_uv_aspect(
    vertices: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Per-vertex ratio tan_len / bit_len where tan_len = |∂pos/∂u|.

    Used by the hatching shader to normalise stripe density so it is the same
    for every direction regardless of UV anisotropy.
    """
    N = len(vertices)
    tan_len_accum  = np.zeros(N, dtype=np.float64)
    btan_len_accum = np.zeros(N, dtype=np.float64)
    count          = np.zeros(N, dtype=np.float64)

    v0  = vertices[indices[:, 0]].astype(np.float64)
    v1  = vertices[indices[:, 1]].astype(np.float64)
    v2  = vertices[indices[:, 2]].astype(np.float64)
    uv0 = uvs[indices[:, 0]].astype(np.float64)
    uv1 = uvs[indices[:, 1]].astype(np.float64)
    uv2 = uvs[indices[:, 2]].astype(np.float64)

    e1  = v1 - v0
    e2  = v2 - v0
    du1 = (uv1 - uv0)[:, 0:1]
    dv1 = (uv1 - uv0)[:, 1:2]
    du2 = (uv2 - uv0)[:, 0:1]
    dv2 = (uv2 - uv0)[:, 1:2]

    denom = du1 * dv2 - du2 * dv1
    r     = np.where(np.abs(denom) > 1e-12, 1.0 / denom, 0.0)

    tan  = r * (dv2 * e1 - dv1 * e2)   # ∂pos/∂u  (F, 3)
    btan = r * (du1 * e2 - du2 * e1)   # ∂pos/∂v  (F, 3)

    tl = np.linalg.norm(tan,  axis=1)  # (F,)
    bl = np.linalg.norm(btan, axis=1)  # (F,)

    for i in range(3):
        np.add.at(tan_len_accum,  indices[:, i], tl)
        np.add.at(btan_len_accum, indices[:, i], bl)
        np.add.at(count,          indices[:, i], 1.0)

    avg_tan  = tan_len_accum  / np.maximum(count, 1.0)
    avg_btan = btan_len_accum / np.maximum(count, 1.0)

    aspect       = avg_tan / np.maximum(avg_btan, 1e-8)
    face_aspect  = tl / np.maximum(bl, 1e-8)
    return (aspect.astype(np.float32), avg_tan.astype(np.float32),
            face_aspect.astype(np.float32), tl.astype(np.float32))
