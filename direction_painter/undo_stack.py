from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class UndoEntry:
    layer_index: int
    region: Tuple[int, int, int, int]   # x, y, w, h  (in texel space)
    snapshot: np.ndarray                # (h, w, 2) float32 — state BEFORE the action


class UndoStack:
    def __init__(self, max_depth: int = 50) -> None:
        self.history: List[UndoEntry] = []
        self.future:  List[UndoEntry] = []
        self.max_depth = max_depth

    def push(self, entry: UndoEntry) -> None:
        self.history.append(entry)
        self.future.clear()
        if len(self.history) > self.max_depth:
            self.history.pop(0)

    def snapshot_region(
        self,
        layer_index: int,
        texture: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> UndoEntry:
        """Create an UndoEntry capturing the current state of *region* in *texture*."""
        x, y, w, h = region
        snap = texture[y : y + h, x : x + w, :].copy()
        return UndoEntry(layer_index=layer_index, region=region, snapshot=snap)

    def undo(self, layers) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Restore the last action.
        Returns (layer_index, region) so the caller can upload the changed region to GPU,
        or None if there is nothing to undo.
        """
        if not self.history:
            return None
        entry = self.history.pop()
        layer = layers[entry.layer_index]
        x, y, w, h = entry.region

        # Capture current state for redo
        redo_snap = layer.texture[y : y + h, x : x + w, :].copy()
        redo_entry = UndoEntry(entry.layer_index, entry.region, redo_snap)
        self.future.append(redo_entry)

        # Restore previous state
        layer.texture[y : y + h, x : x + w, :] = entry.snapshot
        layer.dirty = True
        return entry.layer_index, entry.region

    def redo(self, layers) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Re-apply the last undone action.
        Returns (layer_index, region) or None.
        """
        if not self.future:
            return None
        entry = self.future.pop()
        layer = layers[entry.layer_index]
        x, y, w, h = entry.region

        # Capture current state for undo
        undo_snap = layer.texture[y : y + h, x : x + w, :].copy()
        undo_entry = UndoEntry(entry.layer_index, entry.region, undo_snap)
        self.history.append(undo_entry)

        # Apply redo state
        layer.texture[y : y + h, x : x + w, :] = entry.snapshot
        layer.dirty = True
        return entry.layer_index, entry.region

    def clear(self) -> None:
        self.history.clear()
        self.future.clear()
