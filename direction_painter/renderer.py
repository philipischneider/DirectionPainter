"""
renderer.py — All OpenGL / moderngl rendering for DirectionPainter.

Two main passes:
  1. Picking pass  (offscreen RGBA32F FBO) — encodes triangle_id + barycentric
  2. Viewport pass (offscreen RGBA8 FBO)  — diffuse shading + direction tick overlay

The viewport FBO's colour texture (.glo) is passed to imgui.image() each frame.
"""
from __future__ import annotations
import struct
import numpy as np
import moderngl
from scipy.spatial import cKDTree

from .scene import Mesh, DirectionLayer, Scene
from .camera import OrbitCamera


# ─────────────────────────────────────────────────────────────────────────────
# GLSL sources
# ─────────────────────────────────────────────────────────────────────────────

_PICK_VERT = """
#version 330 core
in vec3  in_position;
in vec2  in_bary_uv;   // (1,0) / (0,1) / (0,0) per vertex
in float in_tri_id;

flat out float v_tri_id;
out vec2       v_bary_uv;

uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_tri_id   = in_tri_id;
    v_bary_uv  = in_bary_uv;
}
"""

_PICK_FRAG = """
#version 330 core
flat in float v_tri_id;
in vec2       v_bary_uv;

out vec4 out_color;

void main() {
    // R = triangle_id (exact float up to 2^24), G = bary_u, B = bary_v
    out_color = vec4(v_tri_id, v_bary_uv.x, v_bary_uv.y, 1.0);
}
"""

_VIEW_VERT = """
#version 330 core
in vec3  in_position;
in vec3  in_normal;
in vec2  in_uv;
in float in_uv_aspect;    // |∂pos/∂u| / |∂pos/∂v| per vertex
in float in_uv_tan_len;   // |∂pos/∂u| in world units per vertex

out vec3  v_normal;
out vec2  v_uv;
out float v_uv_aspect;
out float v_uv_tan_len;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_normal     = normalize(u_normal_mat * in_normal);
    v_uv         = in_uv;
    v_uv_aspect  = in_uv_aspect;
    v_uv_tan_len = in_uv_tan_len;
}
"""

_VIEW_FRAG = """
#version 330 core
in vec3  v_normal;
in vec2  v_uv;
in float v_uv_aspect;    // |∂pos/∂u| / |∂pos/∂v|
in float v_uv_tan_len;   // |∂pos/∂u| in world units

out vec4 out_color;

uniform vec3      u_light_dir;
uniform sampler2D u_dir_tex;
uniform bool      u_has_dir_tex;
uniform bool      u_show_hatching;
uniform float     u_hatch_freq;       // lines per world unit
uniform int       u_flowmap_mode;     // 0 = hue wheel, 1 = raw channels (R/G)

void main() {
    vec3  n          = normalize(v_normal);
    float ndl        = max(dot(n, normalize(u_light_dir)), 0.0);
    float brightness = 0.15 + 0.85 * ndl;

    if (u_show_hatching) {
        // ── Hatching pass ─────────────────────────────────────────────
        // Texture stores (cos θ, sin θ): standard 360° direction encoding.
        // Decode orientation: θ = atan2(G, R).
        vec2  rg   = u_has_dir_tex ? texture(u_dir_tex, v_uv).rg : vec2(1.0, 0.0);
        float a    = atan(rg.g, rg.r);
        vec2  dir  = vec2(cos(a), sin(a));
        vec2  perp = vec2(-dir.y, dir.x);   // perpendicular = stripe-spacing axis

        // Convert u_hatch_freq from "lines per world unit" to a UV-space stripe
        // coordinate that advances by 1 per stripe period.
        //
        // Moving Δ world units along perp corresponds to moving
        //   Δuv = Δ / (tan_len × norm_factor)  UV units along perp,
        // where norm_factor = |perp in world-normalised UV| accounts for
        // anisotropy and tan_len = |∂pos/∂u| is the absolute UV scale.
        // Therefore  s = dot(v_uv, perp) × norm_factor × tan_len × freq
        // gives exactly freq stripe cycles per world unit — uniform for every
        // direction and every part of the mesh.
        float aspect      = max(v_uv_aspect, 1e-4);
        float norm_factor = length(vec2(perp.x, perp.y / aspect));
        float tan_len     = max(v_uv_tan_len, 1e-6);
        float s   = dot(v_uv, perp) * norm_factor * tan_len * u_hatch_freq;
        float fw  = fwidth(s) * 1.5;

        // When fwidth is large the direction is changing faster than the stripe
        // period — the effective local frequency is too high to render clearly.
        // Fade out the hatching in those regions instead of showing aliased
        // thin lines at the boundary between two painted directions.
        float line_width = 0.45;
        float fade = 1.0 - smoothstep(line_width, 1.0, fw);

        float b   = fract(s);
        float ink = (1.0 - smoothstep(line_width - fw, line_width + fw, b)) * fade;

        vec3 paper_col = vec3(0.95, 0.93, 0.88);
        vec3 ink_col   = vec3(0.07, 0.06, 0.05);
        out_color = vec4(mix(paper_col, ink_col, ink), 1.0);

    } else {
        // ── Standard diffuse + direction colour overlay ───────────────
        vec3 base = vec3(0.65, 0.67, 0.72);
        vec3 col  = base * (0.25 + 0.75 * ndl);

        if (u_has_dir_tex) {
            vec2  rg      = texture(u_dir_tex, v_uv).rg;
            float painted = 1.0 - step(0.999, rg.r) * step(-0.001, -abs(rg.g));

            if (u_flowmap_mode == 1) {
                // Raw channels: show the stored (cos θ, sin θ) values directly.
                // Remap each channel from [-1, 1] to [0, 1] so the full range
                // is visible.  R channel → red axis, G channel → green axis.
                vec3 raw = vec3((rg.r + 1.0) * 0.5, (rg.g + 1.0) * 0.5, 0.0);
                col = mix(col, raw, painted);
            } else {
                // Hue wheel: map θ ∈ (-π, π] onto full hue [0, 1].
                float angle  = atan(rg.g, rg.r);
                float h      = fract((angle + 3.14159265) / 6.28318530);
                float sector = fract(h * 6.0);
                vec3  tint;
                int   si = int(h * 6.0) % 6;
                if      (si == 0) tint = vec3(1.0,           sector,        0.0);
                else if (si == 1) tint = vec3(1.0 - sector,  1.0,           0.0);
                else if (si == 2) tint = vec3(0.0,           1.0,           sector);
                else if (si == 3) tint = vec3(0.0,           1.0 - sector,  1.0);
                else if (si == 4) tint = vec3(sector,        0.0,           1.0);
                else              tint = vec3(1.0,           0.0,           1.0 - sector);
                col = mix(col, col * tint, 0.55 * painted);
            }
        }

        out_color = vec4(col, 1.0);
    }
}
"""

_TICK_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_TICK_FRAG = """
#version 330 core
out vec4 out_color;
uniform vec4 u_color;
void main() {
    out_color = u_color;
}
"""

# ── Per-polygon hatching shaders ─────────────────────────────────────────────
# Uses a non-indexed expanded VAO so each triangle has its own flat theta.

_POLY_VERT = """
#version 330 core
in vec3  in_position;
in vec3  in_normal;
in vec2  in_uv;
in float in_uv_aspect;    // per-vertex (same data as VIEW shader)
in float in_uv_tan_len;   // per-vertex (same data as VIEW shader)
in float in_theta;        // per-face hatching direction
in float in_scale;        // per-face density multiplier

out vec3  v_normal;
out vec2  v_uv;
out float v_uv_aspect;    // interpolated — identical to VIEW shader
out float v_uv_tan_len;   // interpolated — identical to VIEW shader
flat out float v_theta;
flat out float v_scale;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

void main() {
    gl_Position  = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_normal     = normalize(u_normal_mat * in_normal);
    v_uv         = in_uv;
    v_uv_aspect  = in_uv_aspect;
    v_uv_tan_len = in_uv_tan_len;
    v_theta      = in_theta;
    v_scale      = in_scale;
}
"""

_POLY_FRAG = """
#version 330 core
in vec3  v_normal;
in vec2  v_uv;
in float v_uv_aspect;    // interpolated per-vertex, same as VIEW shader
in float v_uv_tan_len;   // interpolated per-vertex, same as VIEW shader
flat in float v_theta;
flat in float v_scale;

out vec4 out_color;

uniform vec3  u_light_dir;
uniform bool  u_show_hatching;
uniform float u_hatch_freq;
uniform int   u_flowmap_mode;   // 0 = hue wheel, 1 = raw channels

void main() {
    vec3  n   = normalize(v_normal);
    float ndl = max(dot(n, normalize(u_light_dir)), 0.0);

    if (u_show_hatching) {
        // ── Per-face hatching ──────────────────────────────────────────
        // v_theta and v_scale are flat (constant per face).
        // v_uv_aspect and v_uv_tan_len are interpolated per-vertex, identical
        // to the VIEW shader — this guarantees that after baking, the VIEW
        // shader produces congruent stripes for the same theta.
        vec2  dir  = vec2(cos(v_theta), sin(v_theta));
        vec2  perp = vec2(-dir.y, dir.x);

        float aspect      = max(v_uv_aspect, 1e-4);
        float norm_factor = length(vec2(perp.x, perp.y / aspect));
        float tan_len     = max(v_uv_tan_len, 1e-6);
        // s advances by u_hatch_freq * v_scale lines per world unit,
        // independent of direction (proved by norm_factor * tan_len).
        float s   = dot(v_uv, perp) * norm_factor * tan_len
                    * u_hatch_freq * max(v_scale, 0.01);
        float fw  = fwidth(s) * 1.5;
        float line_width = 0.45;
        float fade = 1.0 - smoothstep(line_width, 1.0, fw);
        float b   = fract(s);
        float ink = (1.0 - smoothstep(line_width - fw, line_width + fw, b)) * fade;

        vec3 paper_col = vec3(0.95, 0.93, 0.88);
        vec3 ink_col   = vec3(0.07, 0.06, 0.05);
        out_color = vec4(mix(paper_col, ink_col, ink), 1.0);

    } else {
        // ── Direction colour overlay (same logic as VIEW shader) ───────
        vec3 base = vec3(0.65, 0.67, 0.72);
        vec3 col  = base * (0.25 + 0.75 * ndl);

        float theta = v_theta;

        if (u_flowmap_mode == 1) {
            // Raw: show (cos θ, sin θ) remapped to [0,1]
            col = vec3(cos(theta) * 0.5 + 0.5,
                       sin(theta) * 0.5 + 0.5,
                       0.0);
        } else {
            // Hue wheel: θ ∈ (-π, π] → full hue revolution
            float h      = fract((theta + 3.14159265) / 6.28318530);
            float sector = fract(h * 6.0);
            vec3  tint;
            int   si = int(h * 6.0) % 6;
            if      (si == 0) tint = vec3(1.0,           sector,        0.0);
            else if (si == 1) tint = vec3(1.0 - sector,  1.0,           0.0);
            else if (si == 2) tint = vec3(0.0,           1.0,           sector);
            else if (si == 3) tint = vec3(0.0,           1.0 - sector,  1.0);
            else if (si == 4) tint = vec3(sector,        0.0,           1.0);
            else              tint = vec3(1.0,           0.0,           1.0 - sector);
            col = mix(col, col * tint, 0.55);
        }

        out_color = vec4(col, 1.0);
    }
}
"""


def _mat4_bytes(m: np.ndarray) -> bytes:
    """Upload a row-major numpy 4×4 to OpenGL column-major bytes."""
    return m.T.astype(np.float32).tobytes()


def _mat3_bytes(m: np.ndarray) -> bytes:
    """Upload a row-major numpy 3×3 to OpenGL column-major bytes."""
    return m.T.astype(np.float32).tobytes()


# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    MAX_TICKS = 8192   # maximum tick line segments

    def __init__(self) -> None:
        self.ctx: moderngl.Context = moderngl.create_context()
        self.ctx.gc_mode = "auto"

        # Shader programs
        self._pick_prog = self.ctx.program(
            vertex_shader=_PICK_VERT,
            fragment_shader=_PICK_FRAG,
        )
        self._view_prog = self.ctx.program(
            vertex_shader=_VIEW_VERT,
            fragment_shader=_VIEW_FRAG,
        )
        self._tick_prog = self.ctx.program(
            vertex_shader=_TICK_VERT,
            fragment_shader=_TICK_FRAG,
        )
        self._poly_prog = self.ctx.program(
            vertex_shader=_POLY_VERT,
            fragment_shader=_POLY_FRAG,
        )

        # FBOs / textures — created on first resize
        self.pick_fbo:  moderngl.Framebuffer | None = None
        self.view_fbo:  moderngl.Framebuffer | None = None
        self.pick_tex:  moderngl.Texture | None     = None
        self.view_tex:  moderngl.Texture | None     = None
        self._fbo_size  = (0, 0)

        # Mesh VAOs — created on upload_mesh
        self._pick_vao:    moderngl.VertexArray | None = None
        self._view_vao:    moderngl.VertexArray | None = None
        self._face_count   = 0

        # Tick VBO / VAO
        self._tick_vbo: moderngl.Buffer = self.ctx.buffer(
            reserve=self.MAX_TICKS * 2 * 12   # 2 vertices × 3 floats × 4 bytes
        )
        self._tick_vao: moderngl.VertexArray = self.ctx.vertex_array(
            self._tick_prog, [(self._tick_vbo, "3f", "in_position")]
        )
        self._tick_count = 0

        # Per-polygon hatching VAO (expanded, non-indexed)
        self._poly_vao:          moderngl.VertexArray | None = None
        self._poly_theta_vbo:    moderngl.Buffer | None      = None
        self._poly_scale_vbo:    moderngl.Buffer | None      = None
        self._poly_face_count    = 0
        self._expanded_face_pos: np.ndarray | None           = None

        # Selected-face highlight (one triangle drawn as fill + outline)
        self._sel_face_vbo = self.ctx.buffer(reserve=3 * 3 * 4)
        self._sel_face_vao = self.ctx.vertex_array(
            self._tick_prog, [(self._sel_face_vbo, "3f", "in_position")]
        )

        # UV → world KD-tree (populated on mesh upload)
        self._uv_tree:        cKDTree | None    = None
        self._face_positions: np.ndarray | None = None
        self._face_tangents:  np.ndarray | None = None
        self._face_bitangents: np.ndarray | None = None

        # Spline preview — curve (LINE_STRIP) and control-point crosses (LINES)
        _MAX_SPLINE = 4096
        _MAX_CTRL   = 128
        self._spline_curve_vbo = self.ctx.buffer(reserve=_MAX_SPLINE * 3 * 4)
        self._spline_curve_vao = self.ctx.vertex_array(
            self._tick_prog, [(self._spline_curve_vbo, "3f", "in_position")]
        )
        self._spline_ctrl_vbo  = self.ctx.buffer(reserve=_MAX_CTRL * 4 * 3 * 4)
        self._spline_ctrl_vao  = self.ctx.vertex_array(
            self._tick_prog, [(self._spline_ctrl_vbo, "3f", "in_position")]
        )
        self._spline_curve_count = 0
        self._spline_ctrl_count  = 0

    # ──────────────────────────────────────────────────────────────────────
    # FBO management
    # ──────────────────────────────────────────────────────────────────────

    def resize_fbos(self, size: tuple[int, int]) -> None:
        w, h = max(1, size[0]), max(1, size[1])
        if (w, h) == self._fbo_size:
            return
        self._fbo_size = (w, h)

        # Release old objects
        for attr in ("pick_fbo", "view_fbo", "pick_tex", "view_tex"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()

        # Picking FBO: RGBA32F
        self.pick_tex = self.ctx.texture((w, h), 4, dtype="f4")
        pick_depth    = self.ctx.depth_renderbuffer((w, h))
        self.pick_fbo = self.ctx.framebuffer([self.pick_tex], pick_depth)

        # Viewport FBO: RGBA8
        self.view_tex  = self.ctx.texture((w, h), 4)
        view_depth     = self.ctx.depth_renderbuffer((w, h))
        self.view_fbo  = self.ctx.framebuffer([self.view_tex], view_depth)

    # ──────────────────────────────────────────────────────────────────────
    # Mesh upload
    # ──────────────────────────────────────────────────────────────────────

    def upload_mesh(self, mesh: Mesh) -> None:
        """Build VAOs from a loaded Mesh."""
        # Release old VAOs if any
        if self._pick_vao:
            self._pick_vao.release()
        if self._view_vao:
            self._view_vao.release()

        # ── Viewport VAO (indexed) ─────────────────────────────────────
        vbo_pos     = self.ctx.buffer(mesh.vertices.tobytes())
        vbo_norm    = self.ctx.buffer(mesh.normals.tobytes())
        vbo_uv      = self.ctx.buffer(mesh.uvs.tobytes())
        vbo_aspect  = self.ctx.buffer(mesh.uv_aspect.astype(np.float32).tobytes())
        vbo_tan_len = self.ctx.buffer(mesh.uv_tan_len.astype(np.float32).tobytes())
        ibo         = self.ctx.buffer(mesh.indices.tobytes())

        self._view_vao = self.ctx.vertex_array(
            self._view_prog,
            [
                (vbo_pos,     "3f", "in_position"),
                (vbo_norm,    "3f", "in_normal"),
                (vbo_uv,      "2f", "in_uv"),
                (vbo_aspect,  "1f", "in_uv_aspect"),
                (vbo_tan_len, "1f", "in_uv_tan_len"),
            ],
            index_buffer=ibo,
            index_element_size=4,
        )

        # ── Picking VAO (non-indexed, expanded per-face) ───────────────
        expanded = _expand_for_picking(mesh)

        vbo_pick_pos   = self.ctx.buffer(expanded["pos"].tobytes())
        vbo_pick_bary  = self.ctx.buffer(expanded["bary"].tobytes())   # (F*3, 2) float32
        vbo_pick_id    = self.ctx.buffer(expanded["ids"].tobytes())    # (F*3,)   float32

        self._pick_vao = self.ctx.vertex_array(
            self._pick_prog,
            [
                (vbo_pick_pos,  "3f", "in_position"),
                (vbo_pick_bary, "2f", "in_bary_uv"),
                (vbo_pick_id,   "1f", "in_tri_id"),
            ],
        )
        self._face_count = mesh.face_count

        # ── UV KD-tree for tick sampling ───────────────────────────────
        if mesh.face_uv_centroids is not None:
            self._uv_tree         = cKDTree(mesh.face_uv_centroids)
            self._face_positions  = mesh.face_pos_centroids
            self._face_tangents   = mesh.face_tangents
            self._face_bitangents = mesh.face_bitangents

        # ── Per-polygon hatching VAO ───────────────────────────────────
        self._build_poly_vao(mesh)

    # ──────────────────────────────────────────────────────────────────────
    # Layer texture management
    # ──────────────────────────────────────────────────────────────────────

    def create_layer_texture(self, layer: DirectionLayer) -> None:
        if layer.gpu_texture is not None:
            layer.gpu_texture.release()
        tex = self.ctx.texture(
            (layer.width, layer.height), 3,
            data=layer.texture.tobytes(), dtype="f4"
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        layer.gpu_texture = tex
        layer.dirty = False

    def upload_dirty_region(
        self,
        layer: DirectionLayer,
        x: int, y: int, w: int, h: int,
    ) -> None:
        if layer.gpu_texture is None:
            self.create_layer_texture(layer)
            return
        region = layer.texture[y : y + h, x : x + w, :]
        layer.gpu_texture.write(region.astype(np.float32).tobytes(),
                                 viewport=(x, y, w, h))
        layer.dirty = False

    # ──────────────────────────────────────────────────────────────────────
    # Per-polygon hatching
    # ──────────────────────────────────────────────────────────────────────

    def _build_poly_vao(self, mesh: Mesh) -> None:
        """Create the expanded (non-indexed) VAO for per-face hatching.

        All normalization attributes (uv_aspect, uv_tan_len) are expanded
        from per-face values so every vertex of a face carries the same value.
        Together with the `flat` qualifier in the shader, this guarantees that
        stripe density is constant across each face and does NOT change when
        the user rotates theta.
        """
        F        = mesh.face_count
        flat_idx = mesh.indices.reshape(-1)   # (F*3,)

        pos   = mesh.vertices[flat_idx].astype(np.float32)
        norms = mesh.normals[flat_idx].astype(np.float32)
        uvs   = mesh.uvs[flat_idx].astype(np.float32)

        self._expanded_face_pos = pos.copy()

        # Use the same per-vertex averaged values as the VIEW shader so that
        # s = dot(v_uv, perp) * norm_factor * tan_len is computed identically
        # in both shaders.  This is what makes baked flow-map hatching and
        # poly-mode hatching look congruent.
        aspects  = mesh.uv_aspect[flat_idx].astype(np.float32)
        tan_lens = mesh.uv_tan_len[flat_idx].astype(np.float32)

        theta = np.zeros(F * 3, dtype=np.float32)   # default: dir=(1,0), perp=(0,1)
        scale = np.ones(F * 3,  dtype=np.float32)   # default scale = 1×

        if self._poly_vao is not None:
            self._poly_vao.release()

        vbo_pos     = self.ctx.buffer(pos.tobytes())
        vbo_norms   = self.ctx.buffer(norms.tobytes())
        vbo_uvs     = self.ctx.buffer(uvs.tobytes())
        vbo_aspects = self.ctx.buffer(aspects.tobytes())
        vbo_tanlens = self.ctx.buffer(tan_lens.tobytes())
        self._poly_theta_vbo = self.ctx.buffer(theta.tobytes())
        self._poly_scale_vbo = self.ctx.buffer(scale.tobytes())

        self._poly_vao = self.ctx.vertex_array(
            self._poly_prog,
            [
                (vbo_pos,              "3f", "in_position"),
                (vbo_norms,            "3f", "in_normal"),
                (vbo_uvs,              "2f", "in_uv"),
                (vbo_aspects,          "1f", "in_uv_aspect"),
                (vbo_tanlens,          "1f", "in_uv_tan_len"),
                (self._poly_theta_vbo, "1f", "in_theta"),
                (self._poly_scale_vbo, "1f", "in_scale"),
            ],
        )
        self._poly_face_count = F

    def update_poly_theta(self, poly_theta: np.ndarray) -> None:
        """Write per-face theta values (F,) to the poly VAO's theta buffer."""
        if self._poly_theta_vbo is None:
            return
        data = np.repeat(poly_theta.astype(np.float32), 3)
        self._poly_theta_vbo.write(data.tobytes())

    def update_poly_scale(self, poly_scale: np.ndarray) -> None:
        """Write per-face scale values (F,) to the poly VAO's scale buffer."""
        if self._poly_scale_vbo is None:
            return
        data = np.repeat(poly_scale.astype(np.float32), 3)
        self._poly_scale_vbo.write(data.tobytes())

    # ──────────────────────────────────────────────────────────────────────
    # Tick overlay
    # ──────────────────────────────────────────────────────────────────────

    def update_ticks(self, mesh: Mesh, layer: DirectionLayer,
                     spacing: int = 16) -> None:
        """Rebuild the GL_LINES tick buffer from the current direction texture."""
        if self._uv_tree is None:
            return

        H, W     = layer.height, layer.width
        tex      = layer.texture
        endpoints: list[np.ndarray] = []

        # Sample a regular UV grid
        for tv_px in range(0, H, spacing):
            for tu_px in range(0, W, spacing):
                r, g = tex[tv_px, tu_px, 0], tex[tv_px, tu_px, 1]
                if r == 1.0 and g == 0.0:
                    continue   # neutral / unpainted — skip

                tu = tu_px / W
                tv = tv_px / H

                # Find nearest face centroid in UV space
                dist, fi = self._uv_tree.query([tu, tv])
                if dist > 0.05:   # no nearby geometry
                    continue

                pos   = self._face_positions[fi]
                tan   = self._face_tangents[fi]
                bitan = self._face_bitangents[fi]

                theta = float(np.arctan2(g, r))   # texture stores θ directly
                d     = np.cos(theta) * tan + np.sin(theta) * bitan
                half  = 0.015   # world-space half-length of tick

                endpoints.append(pos - d * half)
                endpoints.append(pos + d * half)

                if len(endpoints) >= self.MAX_TICKS * 2:
                    break
            if len(endpoints) >= self.MAX_TICKS * 2:
                break

        self._tick_count = len(endpoints)
        if endpoints:
            data = np.array(endpoints, dtype=np.float32).tobytes()
            self._tick_vbo.write(data)

    # ──────────────────────────────────────────────────────────────────────
    # Spline preview
    # ──────────────────────────────────────────────────────────────────────

    def update_spline_preview(
        self,
        curve_world: np.ndarray | None,   # (N, 3) sampled curve in world space
        ctrl_world:  list | None,         # list of (3,) exact world positions
    ) -> None:
        """Upload world-space positions to the spline preview VBOs."""
        # ── Curve (LINE_STRIP) ──────────────────────────────────────────
        if curve_world is not None and len(curve_world) > 0:
            pts = np.asarray(curve_world, dtype=np.float32)
            n   = min(len(pts), 4096)
            self._spline_curve_vbo.write(pts[:n].tobytes())
            self._spline_curve_count = n
        else:
            self._spline_curve_count = 0

        # ── Control point crosses (GL_LINES) ────────────────────────────
        # Use fixed world-space axes for the cross arms — the position is what
        # matters; orientation is purely cosmetic.
        if ctrl_world is not None and len(ctrl_world) > 0:
            half = 0.016
            ax   = np.array([half, 0.0,  0.0], dtype=np.float32)
            ay   = np.array([0.0,  half, 0.0], dtype=np.float32)
            crosses = []
            for pos in ctrl_world:
                p = np.asarray(pos, dtype=np.float32)
                crosses.extend([p - ax, p + ax, p - ay, p + ay])
            m = min(len(crosses), 128 * 4)
            data = np.array(crosses[:m], dtype=np.float32)
            self._spline_ctrl_vbo.write(data.tobytes())
            self._spline_ctrl_count = m
        else:
            self._spline_ctrl_count = 0

    def clear_spline_preview(self) -> None:
        self._spline_curve_count = 0
        self._spline_ctrl_count  = 0

    # ──────────────────────────────────────────────────────────────────────
    # Render passes
    # ──────────────────────────────────────────────────────────────────────

    def render_picking(self, camera: OrbitCamera, size: tuple[int, int]) -> None:
        if self._pick_vao is None or self.pick_fbo is None:
            return

        w, h  = size
        model = np.eye(4, dtype=np.float32)
        V     = camera.view_matrix()
        P     = camera.proj_matrix(w / max(h, 1))
        mvp   = P @ V @ model

        self.pick_fbo.use()
        self.pick_fbo.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self._pick_prog["u_mvp"].write(_mat4_bytes(mvp))
        self._pick_vao.render(moderngl.TRIANGLES)

    def render_viewport(
        self,
        camera: OrbitCamera,
        size: tuple[int, int],
        layer: DirectionLayer | None,
        show_hatching: bool = False,
        hatch_freq: float = 30.0,
        flowmap_mode: int = 0,
        use_poly_theta: bool = False,
        selected_face: int = -1,
    ) -> None:
        if self._view_vao is None or self.view_fbo is None:
            return

        w, h  = size
        model = np.eye(4, dtype=np.float32)
        V     = camera.view_matrix()
        P     = camera.proj_matrix(w / max(h, 1))
        mvp   = P @ V @ model

        normal_mat = np.linalg.inv(model[:3, :3]).T

        self.view_fbo.use()
        self.view_fbo.viewport = (0, 0, w, h)
        self.ctx.clear(0.10, 0.10, 0.11, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        prog = self._view_prog
        prog["u_model"].write(_mat4_bytes(model))
        prog["u_view"].write(_mat4_bytes(V))
        prog["u_proj"].write(_mat4_bytes(P))
        prog["u_normal_mat"].write(_mat3_bytes(normal_mat))
        light = np.array([0.6, 1.0, 0.8], dtype=np.float32)
        light /= np.linalg.norm(light)
        prog["u_light_dir"].value = tuple(light)

        # Bind direction texture (unit 0)
        if layer is not None and layer.gpu_texture is not None:
            layer.gpu_texture.use(location=0)
            prog["u_dir_tex"].value = 0
            prog["u_has_dir_tex"].value = True
        else:
            prog["u_has_dir_tex"].value = False

        if use_poly_theta and self._poly_vao is not None:
            # ── Per-polygon mode: render with poly prog + flat theta ───
            pp = self._poly_prog
            pp["u_model"].write(_mat4_bytes(model))
            pp["u_view"].write(_mat4_bytes(V))
            pp["u_proj"].write(_mat4_bytes(P))
            pp["u_normal_mat"].write(_mat3_bytes(normal_mat))
            pp["u_light_dir"].value     = tuple(light)
            pp["u_show_hatching"].value  = show_hatching
            pp["u_hatch_freq"].value     = hatch_freq
            pp["u_flowmap_mode"].value   = flowmap_mode
            self._poly_vao.render(moderngl.TRIANGLES)
        else:
            # ── Flow-map mode: render with view prog + texture ─────────
            prog["u_show_hatching"].value = show_hatching
            prog["u_hatch_freq"].value    = hatch_freq
            prog["u_flowmap_mode"].value  = flowmap_mode
            self._view_vao.render(moderngl.TRIANGLES)

        # Tick overlay (flow-map mode only)
        if self._tick_count > 0 and not use_poly_theta:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._tick_prog["u_mvp"].write(_mat4_bytes(mvp))
            self._tick_prog["u_color"].value = (1.0, 0.85, 0.2, 1.0)
            self._tick_vao.render(moderngl.LINES, vertices=self._tick_count)

        # Spline preview overlay
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._tick_prog["u_mvp"].write(_mat4_bytes(mvp))
        if self._spline_curve_count > 0:
            self._tick_prog["u_color"].value = (0.3, 0.9, 1.0, 1.0)   # cyan curve
            self._spline_curve_vao.render(moderngl.LINE_STRIP,
                                          vertices=self._spline_curve_count)
        if self._spline_ctrl_count > 0:
            self._tick_prog["u_color"].value = (1.0, 1.0, 1.0, 1.0)   # white crosses
            self._spline_ctrl_vao.render(moderngl.LINES,
                                         vertices=self._spline_ctrl_count)

        # Selected-face highlight (poly mode)
        if (
            use_poly_theta
            and selected_face >= 0
            and self._expanded_face_pos is not None
            and selected_face * 3 + 3 <= len(self._expanded_face_pos)
        ):
            verts = self._expanded_face_pos[selected_face * 3 : selected_face * 3 + 3]
            self._sel_face_vbo.write(verts.tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._tick_prog["u_color"].value = (1.0, 0.85, 0.1, 0.30)
            self._sel_face_vao.render(moderngl.TRIANGLES, vertices=3)
            self._tick_prog["u_color"].value = (1.0, 0.85, 0.1, 1.00)
            self._sel_face_vao.render(moderngl.LINE_LOOP, vertices=3)
            self.ctx.disable(moderngl.BLEND)

        # Restore default framebuffer so Dear ImGui can render
        self.ctx.screen.use()

    # ──────────────────────────────────────────────────────────────────────
    # Picking readback
    # ──────────────────────────────────────────────────────────────────────

    def read_pick_pixel(
        self, x: int, y: int
    ) -> tuple[int, float, float]:
        """
        Read the picking buffer at screen pixel (x, y) — (0,0) = top-left.
        Returns (triangle_id, bary_u, bary_v).  triangle_id == -1 on miss.
        """
        if self.pick_fbo is None:
            return -1, 0.0, 0.0

        W, H  = self._fbo_size
        gl_y  = H - 1 - y   # flip: screen y-down → OpenGL y-up

        if not (0 <= x < W and 0 <= gl_y < H):
            return -1, 0.0, 0.0

        self.pick_fbo.use()
        raw = self.pick_fbo.read(viewport=(x, gl_y, 1, 1), components=4, dtype="f4")
        self.ctx.screen.use()

        r, g, b, a = struct.unpack("4f", raw)

        # triangle_id is stored as a float (exact for ints up to 2^24)
        tri_id = int(round(r))
        if tri_id < 0 or tri_id >= self._face_count:
            return -1, 0.0, 0.0

        return tri_id, float(g), float(b)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expand_for_picking(mesh: Mesh) -> dict:
    """
    Build a non-indexed (expanded) vertex buffer for the picking pass.
    Each triangle occupies 3 consecutive vertices with:
      - positions from the original mesh
      - barycentric coords (1,0), (0,1), (0,0) per vertex
      - triangle id repeated 3×
    """
    F = mesh.face_count
    flat_idx = mesh.indices.reshape(-1)                   # (F*3,)

    pos  = mesh.vertices[flat_idx].astype(np.float32)     # (F*3, 3)

    # Barycentric: for vertex 0 in each face → (1,0), vertex 1 → (0,1), vertex 2 → (0,0)
    bary_pattern = np.tile(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        (F, 1)
    )                                                      # (F*3, 2)

    ids = np.repeat(np.arange(F, dtype=np.float32), 3)    # (F*3,)

    return {"pos": pos, "bary": bary_pattern, "ids": ids}
