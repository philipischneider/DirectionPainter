"""
app.py — Main application using hello_imgui (from imgui-bundle).

Layout:
  ┌───────────────────────┬──────────────┐
  │  Viewport (3D)        │  Side panel  │
  └───────────────────────┴──────────────┘

The 3D viewport is rendered to a moderngl FBO and displayed via imgui.image().
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from imgui_bundle import hello_imgui, imgui

from .scene import Scene, DirectionLayer
from .camera import OrbitCamera
from .mesh_loader import load_gltf
from .renderer import Renderer
from .painter import Painter
from .spline_painter import SplinePainter
from .undo_stack import UndoStack
from .fill_island import fill_island
from .export import export_layer


PANEL_WIDTH = 300   # pixels


class App:
    def __init__(self) -> None:
        self.scene    = Scene()
        self.camera   = OrbitCamera()
        self.undo     = UndoStack()
        self.renderer:       Optional[Renderer]      = None
        self.painter:        Optional[Painter]        = None
        self.spline_painter: Optional[SplinePainter] = None

        self._tool          = "Brush"   # "Brush" | "Spline"
        self._is_painting   = False
        self._is_navigating = False

        self._viewport_pos  = (0, 0)   # screen position of the viewport image
        self._viewport_size = (800, 600)

        self._tex_resolution = 1024

        self._show_hatching = False
        self._hatch_freq    = 30.0

        # Status messages shown at the bottom of the side panel
        self._status = ""

    # ──────────────────────────────────────────────────────────────────────
    # hello_imgui entry-point
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "DirectionPainter"
        params.app_window_params.window_geometry.size = (1280, 800)
        params.fps_idling.enable_idling = False   # continuous refresh for viewport

        params.callbacks.post_init    = self._post_init
        params.callbacks.show_gui     = self._show_gui
        params.callbacks.before_exit  = self._before_exit

        hello_imgui.run(params)

    # ──────────────────────────────────────────────────────────────────────
    # hello_imgui callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _post_init(self) -> None:
        """Called once after the OpenGL context is created."""
        self.renderer       = Renderer()
        self.painter        = Painter(self.scene, self.undo)
        self.spline_painter = SplinePainter(self.scene)

    def _before_exit(self) -> None:
        pass   # moderngl objects will be released by GC

    def _show_gui(self) -> None:
        io = imgui.get_io()
        W  = int(io.display_size.x)
        H  = int(io.display_size.y)
        vp_w = max(1, W - PANEL_WIDTH)

        # ── Viewport window ────────────────────────────────────────────
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((float(vp_w), float(H)))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0.0, 0.0))
        imgui.begin("##viewport", None, flags)
        imgui.pop_style_var()

        avail = imgui.get_content_region_avail()
        vw, vh = max(1, int(avail.x)), max(1, int(avail.y))

        if self.renderer is not None:
            self.renderer.resize_fbos((vw, vh))

            if self.scene.mesh is not None:
                self.renderer.render_picking(self.camera, (vw, vh))
                self.renderer.render_viewport(
                    self.camera, (vw, vh), self.scene.active_layer,
                    show_hatching=self._show_hatching,
                    hatch_freq=self._hatch_freq,
                )

            if self.renderer.view_tex is not None:
                # Display the viewport FBO texture (flip Y: uv0=(0,1), uv1=(1,0))
                # imgui-bundle requires ImTextureRef for texture IDs, not raw ints
                tex_ref = imgui.ImTextureRef(self.renderer.view_tex.glo)
                imgui.image(
                    tex_ref,
                    (float(vw), float(vh)),
                    (0.0, 1.0),   # uv0 — positional to avoid tuple/kwarg mismatch
                    (1.0, 0.0),   # uv1
                )
                # Record the actual screen position of the image for pick coords
                self._viewport_pos  = imgui.get_item_rect_min()
                self._viewport_size = (vw, vh)

        imgui.end()

        # ── Side panel ─────────────────────────────────────────────────
        panel_flags = (
            imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
        )
        imgui.set_next_window_pos((float(vp_w), 0.0))
        imgui.set_next_window_size((float(PANEL_WIDTH), float(H)))
        imgui.begin("Panel##side", None, panel_flags)
        self._draw_panel()
        imgui.end()

        # ── Keyboard shortcuts ─────────────────────────────────────────
        self._handle_keys(io)

        # ── Mouse events in viewport ───────────────────────────────────
        self._handle_mouse(io, vw, vh)

    # ──────────────────────────────────────────────────────────────────────
    # Panel drawing
    # ──────────────────────────────────────────────────────────────────────

    def _draw_panel(self) -> None:
        # File
        imgui.separator_text("File")
        if imgui.button("Open GLTF / GLB..."):
            self._open_file_dialog()
        imgui.same_line()
        if imgui.button("Export Layer...") and self.scene.layers:
            self._export_layer_dialog()

        # Layers
        imgui.separator_text("Layers")
        if imgui.button("+##layer"):
            self._add_layer()
        imgui.same_line()
        if imgui.button("-##layer") and len(self.scene.layers) > 1:
            self._remove_layer()

        for i, layer in enumerate(self.scene.layers):
            selected = (i == self.scene.active_layer_idx)
            ok, _ = imgui.selectable(f"{layer.name}##l{i}", selected)
            if ok:
                self.scene.active_layer_idx = i

        # Tool selector
        imgui.separator_text("Tool")
        for tool in ("Brush", "Spline"):
            active = self._tool == tool
            if active:
                imgui.push_style_color(imgui.Col_.button,
                                       (0.20, 0.55, 0.85, 1.0))
            if imgui.button(f"{tool}##tool"):
                if self._tool != tool:
                    self._switch_tool(tool)
            if active:
                imgui.pop_style_color()
            imgui.same_line()
        imgui.new_line()

        # Tool-specific settings
        if self._tool == "Brush":
            self._draw_brush_panel()
        else:
            self._draw_spline_panel()

        # Hatching preview
        imgui.separator_text("Hatching")
        label = "Hatching ON" if self._show_hatching else "Hatching OFF"
        color = (0.15, 0.55, 0.85, 1.0) if self._show_hatching else (0.35, 0.35, 0.35, 1.0)
        imgui.push_style_color(imgui.Col_.button, color)
        if imgui.button(label, (-1.0, 0.0)):
            self._show_hatching = not self._show_hatching
        imgui.pop_style_color()
        if self._show_hatching:
            _, self._hatch_freq = imgui.slider_float(
                "Frequency", self._hatch_freq, 5.0, 120.0)

        # Fill Island
        imgui.separator_text("Fill")
        if imgui.button("Fill Island", (-1.0, 0.0)):
            self._do_fill_island()

        # Texture Resolution
        imgui.separator_text("Texture Resolution")
        for res in [512, 1024, 2048]:
            label = f"{'>' if res == self._tex_resolution else ' '}{res}"
            if imgui.button(label):
                self._set_texture_resolution(res)
            imgui.same_line()
        imgui.new_line()

        # Status
        if self._status:
            imgui.separator()
            imgui.text_wrapped(self._status)

    def _draw_brush_panel(self) -> None:
        imgui.separator_text("Brush")
        modes = ["Stroke", "Radial", "Circular", "Comb"]
        if self.painter:
            current = modes.index(self.painter.brush.mode)
            changed, current = imgui.combo("Mode", current, modes)
            if changed:
                self.painter.brush.mode = modes[current]
            _, self.painter.brush.radius   = imgui.slider_float(
                "Radius",   self.painter.brush.radius,   0.005, 0.3)
            _, self.painter.brush.strength = imgui.slider_float(
                "Strength", self.painter.brush.strength, 0.01,  1.0)
            if self.painter.brush.mode == "Radial":
                _, self.painter.brush.inward = imgui.checkbox(
                    "Inward", self.painter.brush.inward)

    def _draw_spline_panel(self) -> None:
        imgui.separator_text("Spline")
        sp = self.spline_painter
        if sp is None:
            return

        _, sp.params.radius   = imgui.slider_float(
            "Radius##sp",   sp.params.radius,   0.005, 0.3)
        _, sp.params.strength = imgui.slider_float(
            "Strength##sp", sp.params.strength, 0.01,  1.0)

        n = len(sp.ctrl_uvs)
        if n == 0:
            imgui.text_disabled("Click on the mesh to place points.")
        else:
            imgui.text(f"{n} point{'s' if n != 1 else ''} placed.")
            imgui.text_disabled("Enter — commit   Esc — cancel")
            imgui.spacing()
            if imgui.button("Commit##sp", (-1.0, 0.0)):
                self._spline_commit()
            if imgui.button("Cancel##sp", (-1.0, 0.0)):
                self._spline_cancel()

    # ──────────────────────────────────────────────────────────────────────
    # Mouse handling
    # ──────────────────────────────────────────────────────────────────────

    def _handle_mouse(self, io: imgui.IO, vw: int, vh: int) -> None:
        mp = io.mouse_pos
        vx = int(self._viewport_pos[0]) if hasattr(self._viewport_pos, '__getitem__') else int(self._viewport_pos.x)
        vy = int(self._viewport_pos[1]) if hasattr(self._viewport_pos, '__getitem__') else int(self._viewport_pos.y)

        lx = int(mp.x) - vx
        ly = int(mp.y) - vy
        in_vp = 0 <= lx < vw and 0 <= ly < vh

        lmb_clicked  = imgui.is_mouse_clicked(imgui.MouseButton_.left)
        lmb_down     = io.mouse_down[0]
        lmb_released = imgui.is_mouse_released(imgui.MouseButton_.left)
        mmb_down     = io.mouse_down[2]
        mmb_released = imgui.is_mouse_released(imgui.MouseButton_.middle)
        dx = float(io.mouse_delta.x)
        dy = float(io.mouse_delta.y)

        # Zoom (scroll wheel)
        if in_vp and io.mouse_wheel != 0.0:
            self.camera.zoom(float(io.mouse_wheel))

        # Orbit / Pan — MMB drag (Blender standard)
        # Navigation continues outside the viewport once started, so a fast
        # drag that leaves the window border does not snap the camera.
        if mmb_down and (in_vp or self._is_navigating):
            self._is_navigating = True
            if io.key_shift:
                self.camera.pan(dx, dy)
            else:
                self.camera.orbit(dx * 0.005, dy * 0.005)
        if mmb_released:
            self._is_navigating = False

        # ── Brush tool ─────────────────────────────────────────────────
        if self._tool == "Brush":
            if (
                in_vp and lmb_clicked
                and self.scene.mesh is not None
                and self.scene.layers
                and self.renderer is not None
                and self.painter is not None
            ):
                tri, u, v = self.renderer.read_pick_pixel(lx, ly)
                if tri >= 0:
                    self._is_painting = True
                    self.painter.begin_stroke(tri, u, v, self.camera,
                                              (lx, ly), (vw, vh))

            if lmb_down and self._is_painting:
                if self.renderer is not None and self.painter is not None:
                    tri, u, v = self.renderer.read_pick_pixel(lx, ly)
                    if tri >= 0:
                        self.painter.paint_point(tri, u, v, self.camera,
                                                 (lx, ly), (vw, vh))

            if lmb_released and self._is_painting:
                if self.painter is not None and self.renderer is not None:
                    self.painter.end_stroke(self.renderer)
                self._is_painting = False

        # ── Spline tool ────────────────────────────────────────────────
        elif self._tool == "Spline":
            if (
                in_vp and lmb_clicked
                and self.scene.mesh is not None
                and self.scene.layers
                and self.renderer is not None
                and self.spline_painter is not None
                and self.painter is not None
            ):
                tri, u, v = self.renderer.read_pick_pixel(lx, ly)
                if tri >= 0:
                    uv        = self.painter._interpolate_uv(tri, u, v)
                    world_pos = self.painter._interpolate_attr(
                        tri, u, v, self.scene.mesh.vertices)
                    self.spline_painter.add_point(uv, world_pos)
                    self._refresh_spline_preview()

    # ──────────────────────────────────────────────────────────────────────
    # Keyboard handling
    # ──────────────────────────────────────────────────────────────────────

    def _handle_keys(self, io: imgui.IO) -> None:
        if imgui.is_any_item_active():
            return   # don't steal keys from input widgets

        ctrl = io.key_ctrl
        if ctrl and imgui.is_key_pressed(imgui.Key.z):
            self._do_undo()
        if ctrl and imgui.is_key_pressed(imgui.Key.y):
            self._do_redo()

        if self._tool == "Spline":
            if imgui.is_key_pressed(imgui.Key.enter):
                self._spline_commit()
            if imgui.is_key_pressed(imgui.Key.escape):
                self._spline_cancel()

    # ──────────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────────

    def _switch_tool(self, tool: str) -> None:
        if self._tool == "Spline" and self.spline_painter is not None:
            self.spline_painter.cancel()
            if self.renderer is not None:
                self.renderer.clear_spline_preview()
        self._tool = tool

    def _spline_commit(self) -> None:
        if self.spline_painter is None or self.renderer is None:
            return
        if not self.spline_painter.has_points:
            return
        self.spline_painter.commit(self.renderer)
        self.renderer.clear_spline_preview()
        self._status = "Spline committed."

    def _spline_cancel(self) -> None:
        if self.spline_painter is not None:
            self.spline_painter.cancel()
        if self.renderer is not None:
            self.renderer.clear_spline_preview()

    def _refresh_spline_preview(self) -> None:
        if self.spline_painter is None or self.renderer is None:
            return
        self.renderer.update_spline_preview(
            self.spline_painter.preview_world,
            self.spline_painter.ctrl_world,
        )

    def _open_file_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Open GLTF / GLB",
                filetypes=[("GLTF/GLB", "*.gltf *.glb"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception:
            path = ""

        if path:
            self._load_mesh(path)

    def _export_layer_dialog(self) -> None:
        layer = self.scene.active_layer
        if layer is None:
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Export direction layer",
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                initialfile=f"{layer.name}.png",
            )
            root.destroy()
        except Exception:
            path = ""

        if path:
            try:
                export_layer(layer, path)
                self._status = f"Exported: {path}"
            except Exception as e:
                self._status = f"Export failed: {e}"

    def _load_mesh(self, path: str) -> None:
        try:
            mesh = load_gltf(path)
        except Exception as e:
            self._status = f"Load failed: {e}"
            return

        self.scene.mesh = mesh
        self.camera.fit_to_mesh(mesh.vertices)

        # Create a default layer at the current resolution
        layer = DirectionLayer.create("Layer 0", self._tex_resolution,
                                      self._tex_resolution)
        self.scene.layers = [layer]
        self.scene.active_layer_idx = 0

        self.undo.clear()
        self.painter        = Painter(self.scene, self.undo)
        self.spline_painter = SplinePainter(self.scene)

        if self.renderer is not None:
            self.renderer.upload_mesh(mesh)
            self.renderer.create_layer_texture(layer)

        self._status = f"Loaded: {path.split('/')[-1].split(chr(92))[-1]}"

    def _add_layer(self) -> None:
        if self.scene.mesh is None:
            return
        name  = f"Layer {len(self.scene.layers)}"
        layer = DirectionLayer.create(name, self._tex_resolution,
                                      self._tex_resolution)
        self.scene.layers.append(layer)
        self.scene.active_layer_idx = len(self.scene.layers) - 1
        if self.renderer is not None:
            self.renderer.create_layer_texture(layer)

    def _remove_layer(self) -> None:
        if len(self.scene.layers) <= 1:
            return
        idx = self.scene.active_layer_idx
        self.scene.layers.pop(idx)
        self.scene.active_layer_idx = max(0, idx - 1)

    def _do_fill_island(self) -> None:
        if self.scene.mesh is None or not self.scene.layers:
            return
        layer = self.scene.active_layer
        if layer is None:
            return

        # Painted pixels: any texel that differs from the neutral (1, 0)
        neutral = (layer.texture[..., 0] == 1.0) & (layer.texture[..., 1] == 0.0)
        painted = ~neutral

        # Full-layer undo snapshot
        entry = self.undo.snapshot_region(
            self.scene.active_layer_idx,
            layer.texture,
            (0, 0, layer.width, layer.height),
        )
        self.undo.push(entry)

        fill_island(layer, self.scene.mesh, painted)

        if self.renderer is not None:
            self.renderer.create_layer_texture(layer)
            self.renderer.update_ticks(self.scene.mesh, layer)

        self._status = "Fill Island complete."

    def _set_texture_resolution(self, res: int) -> None:
        self._tex_resolution = res
        # Recreate all layers at the new resolution
        for i, layer in enumerate(self.scene.layers):
            new_layer = DirectionLayer.create(layer.name, res, res)
            self.scene.layers[i] = new_layer
            if self.renderer is not None:
                self.renderer.create_layer_texture(new_layer)
        self.undo.clear()
        self._status = f"Texture resolution set to {res}×{res}."

    def _do_undo(self) -> None:
        result = self.undo.undo(self.scene.layers)
        if result and self.renderer is not None:
            li, (x, y, w, h) = result
            self.renderer.upload_dirty_region(self.scene.layers[li], x, y, w, h)
            if self.scene.mesh:
                self.renderer.update_ticks(self.scene.mesh, self.scene.layers[li])

    def _do_redo(self) -> None:
        result = self.undo.redo(self.scene.layers)
        if result and self.renderer is not None:
            li, (x, y, w, h) = result
            self.renderer.upload_dirty_region(self.scene.layers[li], x, y, w, h)
            if self.scene.mesh:
                self.renderer.update_ticks(self.scene.mesh, self.scene.layers[li])
