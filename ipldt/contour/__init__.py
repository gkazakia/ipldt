"""Scanco contour (GOBJ) handling: rendering a mask raster the way IPL renders its
/togobj_from_aim -curvature_smooth 1 contours back into voxels (render_volume / render_slice), the stored
chains themselves (slice_chains, chain_codes) and the GOBJ file reader (read_gobj, gobj_path)."""
from .render import render_volume, render_slice, slice_chains
from .gobj_file import read_gobj, path as gobj_path, chain_codes

__all__ = ["render_volume", "render_slice", "slice_chains", "read_gobj", "gobj_path", "chain_codes"]
