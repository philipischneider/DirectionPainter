# DirectionPainter — Project Specification

## Goal

A standalone desktop application (Python) for painting direction maps onto 3D meshes.
The painted maps will be used in Blender to control the direction of hatching lines
on surfaces, mimicking the engraving aesthetic of Gustave Doré.

This tool focuses exclusively on **painting the direction map**. The Blender hatching
shader is a separate, future concern.

---

## Concept: Direction Map

A direction map is a texture with RG float channels storing a 2D unit vector per texel:

- `R = cos(θ)`, `G = sin(θ)`
- `θ` is the hatch direction angle in UV tangent space
- Analogous to a normal map, but for directions parallel to the surface
- Unpainted regions can default to a neutral value (e.g. `R=1, G=0` → along +U axis)

Multiple direction maps per mesh are supported — one per hatch layer. Each layer is
painted and exported independently as a PNG (RG channels, linear color space).

---

## Input Format: GLTF / GLB

Meshes are exported from Blender as GLTF/GLB with the following data per vertex:

- Position (XYZ)
- Normal (XYZ)
- Tangent (XYZW) — precomputed by Blender using MikkTSpace
- UV coordinates (at least one UV channel)

Bitangent is reconstructed from: `bitangent = cross(normal, tangent.xyz) * tangent.w`

Library: `trimesh` (reads GLTF natively and exposes all vertex attributes).

---

## Tech Stack

| Library | Role |
|---|---|
| `trimesh` | Mesh loading (GLTF/GLB), geometry data access |
| `moderngl` | OpenGL rendering (offscreen + viewport) |
| `imgui-bundle` | Application UI (Dear ImGui with OpenGL backend) |
| `numpy` | All vector/matrix math |
| `scipy` | `ndimage.convolve` for Fill Island diffusion |
| `Pillow` | Texture export (PNG) |

Python 3.10+. No web platform. Desktop only.

---

## Application Architecture

### Data Model

```
Scene
├── mesh: Mesh
│   ├── vertices: np.ndarray (N, 3)  float32
│   ├── normals:  np.ndarray (N, 3)  float32
│   ├── tangents: np.ndarray (N, 4)  float32  (w = handedness)
│   ├── uvs:      np.ndarray (N, 2)  float32
│   └── indices:  np.ndarray (F, 3)  uint32
└── layers: List[DirectionLayer]
    └── DirectionLayer
        ├── name: str
        ├── texture: np.ndarray (H, W, 2)  float32  ← RG direction map
        └── gpu_texture: moderngl.Texture
```

### Render Passes

**Pass 1 — GPU Picking (offscreen framebuffer)**

Encodes per-pixel:
- `R` (16 bits): triangle index high
- `G` (16 bits): triangle index low
- `B` (8 bits): barycentric U (normalized to 0–255)
- `A` (8 bits): barycentric V (normalized to 0–255)

On mouse event: `glReadPixels()` at cursor → decode triangle ID + barycentric coords →
interpolate UV and tangent frame at hit point.

**Pass 2 — Viewport**

- Mesh rendered with simple diffuse shading (Lambert or matcap)
- Overlay of direction ticks: short line segments sampled across the surface,
  oriented by the painted direction. Rendered as `GL_LINES`.
  Ticks only shown where the direction map has been painted.

### Painting Event (mouse drag)

```
1. Read picking buffer at cursor position
2. Decode: triangle_id, bary_u, bary_v
3. Retrieve triangle vertices and attributes
4. Interpolate at barycentric coords:
     uv_hit       = bary @ [uv0, uv1, uv2]
     tangent_hit  = normalize(bary @ [t0, t1, t2])
     bitangent_hit = normalize(bary @ [b0, b1, b2])
5. Compute stroke direction in world space:
     delta = current_cursor_pos - previous_cursor_pos  (screen space)
     unproject both points to world space rays, compute world-space delta on mesh plane
6. Transform world-space stroke direction → tangent space:
     dir_tangent = dot(stroke_world, tangent_hit)
     dir_bitan   = dot(stroke_world, bitangent_hit)
     angle       = atan2(dir_bitan, dir_tangent)
7. Paint to texture:
     For each texel within brush_radius of uv_hit (in UV space):
         weight = brush_falloff(distance)
         existing_angle = atan2(G, R)  [of current texel]
         blended_angle  = angle_slerp(existing_angle, angle, weight)
         texel = (cos(blended_angle), sin(blended_angle))
8. Upload dirty region to GPU texture (glTexSubImage2D)
```

Note: angle blending must use shortest-arc interpolation, not linear RGB blend.

---

## Direction Tick Overlay (Visualization)

To visualize the painted direction without a full hatching shader:

1. Sample a sparse regular grid of points in UV space (e.g. every 16 texels)
2. For each sample:
   - Read direction from texture: `(r, g)` → `θ = atan2(g, r)`
   - Find 3D position on mesh corresponding to that UV point (reverse UV lookup)
   - Compute two endpoints: `p ± (tangent*cos(θ) + bitangent*sin(θ)) * tick_half_length`
3. Render all tick segments as `GL_LINES`

Ticks in unpainted regions are either hidden or shown in a neutral color.

---

## UI Layout (Dear ImGui)

```
┌─────────────────────────────┬──────────────────────────┐
│  Viewport (3D, OpenGL)      │  Side Panel              │
│                             │  ─────────────────────   │
│  [painted mesh with ticks]  │  File                    │
│                             │    [Open GLTF/GLB]       │
│                             │    [Export Layer]        │
│                             │                          │
│                             │  Layers                  │
│                             │    [+] [-] layer list    │
│                             │    active layer name     │
│                             │                          │
│                             │  Brush                   │
│                             │    Mode [Stroke▼]        │
│                             │    Radius  [slider]      │
│                             │    Strength [slider]     │
│                             │                          │
│                             │  Fill                    │
│                             │    [Fill Island]  ←button│
│                             │                          │
│                             │  Texture Resolution      │
│                             │    [512 / 1024 / 2048]   │
└─────────────────────────────┴──────────────────────────┘
```

Viewport navigation:
- Left drag (no paint mode): orbit camera
- Middle drag: pan
- Scroll: zoom
- Right click / dedicated button: toggle paint mode ON/OFF
- Left drag (paint mode): paint stroke

---

## Export Format

Each layer exports as a **PNG with 16-bit precision per channel** (if Pillow supports)
or 8-bit as fallback:

- Channel R: `cos(θ)` remapped from [-1, 1] → [0, 255]
- Channel G: `sin(θ)` remapped from [-1, 1] → [0, 255]
- Channel B: unused (0)
- Linear color space (no gamma correction)

Same convention as a normal map, but only RG channels carry data. This makes it
directly importable into Blender as a texture node with Vector output.

---

## Brush Modes

The tool supports multiple brush types. Each mode determines how the painted direction
is computed at each texel — independently of the stroke direction.

### Stroke (default)
Direction = direction of mouse movement projected onto the surface.
Standard freehand painting for uniform or flowing regions.

### Radial
User clicks to define a center point on the surface; the brush paints directions
pointing outward from (or inward toward) that center.

```
# For each texel within brush radius:
delta = texel_uv - center_uv
dir = normalize(delta)          # outward
dir = normalize(-delta)         # inward variant
```

### Circular
Same center-point interaction as Radial, but direction is tangent to circles
around the center (perpendicular to the radial direction).

```
delta = texel_uv - center_uv
dir = normalize(perp(delta))    # perp = rotate 90°: (-y, x)
                                # CW or CCW depending on sign
```

### Comb (smooth/relax)
Does not set a new direction — instead, blends each texel toward the weighted
average of its UV-space neighbors. Used to smooth out discontinuities and
soften transitions between painted regions.

### Future modes (post-MVP)
- **Singularities / vector field gizmos**: user places source, sink, or vortex
  points on the surface; the direction field is computed analytically from the
  sum of influences. Allows precise, fully editable radial and circular patterns
  without brush painting. Each singularity has position (UV), type, radius, and
  falloff parameters and can be repositioned at any time.

---

## Fill Island (Harmonic Interpolation)

A user-triggered operation (not automatic) that takes the sparse painted directions
within a UV island and smoothly interpolates them across all unpainted texels of that
island.

**Use case:** paint two strokes on a region (e.g. top and bottom of a plane) then
invoke Fill Island to propagate and blend those directions across the entire island.

### Interaction

- User selects the target UV island (click on surface, or pick from list)
- User presses "Fill Island" button in the side panel
- The operation runs and updates the direction texture
- Painted (fixed) pixels are never overwritten — they remain as anchors

### Algorithm: Multi-scale Iterative Diffusion (Jacobi)

Works on the RG channels treated as a complex unit vector `z = R + iG = e^(iθ)`.
Averaging in complex space handles angle wrap-around correctly.

**Step 1 — Build UV island mask**

Rasterize the mesh's UV triangles into a boolean mask at the texture resolution.
Each texel is marked as belonging to a UV island or as empty space.
This mask is computed once on mesh load and cached.

```
For each UV triangle:
    rasterize triangle into island_mask (scan-line fill)
    mark covered texels with island_id
```

**Step 2 — Multi-scale diffusion**

```
scales = [texture_size // 16, // 8, // 4, // 2, full_size]

For each scale:
    downsample field and masks to current scale

    For n_iterations:
        smoothed = convolve2d(field, 3×3 avg kernel)
        field[unpainted & in_island] = smoothed[unpainted & in_island]
        # painted pixels remain fixed (boundary conditions)

    upsample field to next scale (use as warm start)

Upload final field to GPU texture
```

Operates on R and G channels independently (equivalent to operating on cos θ and
sin θ, then normalizing the result).

After diffusion, normalize each texel: `(R, G) = normalize(R, G)` to ensure the
output remains a unit vector.

### Dependencies added

| Library | Role |
|---|---|
| `scipy` | `scipy.ndimage.convolve` for fast convolution in diffusion loop |

---

## References

**flow_map_painter** (ClemensBeute, GitHub):
https://github.com/ClemensBeute/flow_map_painter

A Blender addon that solves the same core problem: painting flow/direction maps
by encoding stroke direction as RGB color `(dir + 1) * 0.5`. Useful as an
algorithm reference, particularly for the UV-space ray cast pipeline
(ray → hit point → barycentric transform → UV interpolation).

Architectural differences that motivate this standalone project:
- No direction tick overlay (no visual feedback of the painted field)
- No layer management
- Triangulates the full mesh on CPU on activation (slow for high-poly)
- Deprecated for Blender 5.0; tightly coupled to Blender's paint API
- No angle-aware blending (writes raw color, no shortest-arc interpolation)

---

## Undo / Redo

A linear undo/redo stack per layer. Each undoable action stores only the **dirty
region** of the texture (not the full texture), keeping memory usage low.

### Undoable actions
- Any paint stroke (Stroke, Radial, Circular, Comb)
- Fill Island

### Data structure

```
UndoStack
├── history: List[UndoEntry]   # past states, oldest first
├── future:  List[UndoEntry]   # redoable states
└── max_depth: int             # e.g. 50 entries

UndoEntry
├── layer_index: int
├── region: (x, y, w, h)      # bounding box of affected texels in UV space
└── snapshot: np.ndarray       # copy of field[y:y+h, x:x+w] before the action
```

### Behaviour

- **Before any paint action or Fill Island:** capture the bounding box of the
  affected region and snapshot the current texel values into a new `UndoEntry`.
  Push onto `history`. Clear `future`.
- **Ctrl+Z (undo):** pop from `history`, swap current region with snapshot,
  push inverse onto `future`. Update GPU texture.
- **Ctrl+Y (redo):** pop from `future`, apply, push onto `history`.
- **Stack overflow:** when `len(history) > max_depth`, discard the oldest entry.

---

## Out of Scope (for now)

- Hatching shader / rendering in Blender
- UV unwrapping (mesh comes with UVs from Blender)
- Multi-mesh scenes
- Symmetry painting
- Singularity/vector field gizmos (post-MVP)
