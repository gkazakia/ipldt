"""ipldt -- an open-source reimplementation of Scanco IPL's distance-transform
morphometry: /dt_thickness, /dt_spacing and /dt_number (Hildebrand & Rüegsegger 1997
sphere fitting as IPL V5.42 implements it), plus IPL's contour (GOBJ) rendering and the
Script 32 / 33 STEP 1 cortical / trabecular separation (ipldt.step1, built on the reimplemented
IPL commands in ipldt.ipl_ops).  Agreement with IPL V5.42 is reported voxel for voxel in the paper.

    from ipldt import dt_thickness, dt_spacing, dt_number, read_aim, align_to, render_volume

    seg  = read_aim("X2420448_SEG_decompressed.AIM")            # cortical 127 + trabecular 126
    trab = read_aim("X2420448_TRAB_SEG_decompressed.AIM")       # on the SEG grid
    mask = read_aim("X2420448_TRAB_MASK_decompressed.AIM")      # raw mask raster, on its own grid
    G = align_to(dict(data=render_volume(mask["data"] > 0).astype("uint8"), dim=mask["dim"], pos=mask["pos"]),
                 seg["dim"], seg["pos"]) > 0                    # IPL's rendering of the contour, on the SEG grid
    th = dt_thickness(trab["data"] > 0, gobj={"rendered": G}, voxel_size_mm=0.0607)
    sp = dt_spacing(seg["data"] > 0, gobj={"rendered": G}, voxel_size_mm=0.0607)
    n  = dt_number(seg["data"] > 0, gobj={"rendered": G}, voxel_size_mm=0.0607)
    th.map, th.report["Th_mm"], sp.report["Sp_mm"], n.report["Tb_N_per_mm"]

    from ipldt import cort_trab_separation, TIBIA
    grey = read_aim("X2420448.AIM")                               # native short greyscale
    per = read_aim("X2420448_T16_00_ALL.AIM")                     # rendered periosteal contour
    r = cort_trab_separation(grey, per, TIBIA)                     # r["cort"], r["trab"] = CORT/TRAB_MASK

Parameters are IPL's: ridge_epsilon (0.9), assign_epsilon (0.5), peel_iter (-1),
version (3); see ipldt.core for what each one does.
"""
from .core import DTResult, diameters, draw_spheres, dt_number, dt_spacing, dt_thickness, peel_gobj, ridge, statistics, surface_distance, surface_vector
from .field import sir_quad
from .contour import render_volume, read_gobj
from .io import align_to, read_aim, write_aim, write_map, write_nifti
from .step1 import RADIUS, TIBIA, Step1Params, cort_trab_separation, cort_trab_separation_from_raw
from . import ipl_ops
from . import porosity

__version__ = "1.0.0"
__all__ = ["dt_thickness", "dt_spacing", "dt_number", "DTResult", "sir_quad", "ridge", "diameters", "draw_spheres",
           "peel_gobj", "statistics", "surface_distance", "surface_vector", "render_volume", "read_gobj",
           "read_aim", "align_to", "write_aim", "write_map", "write_nifti", "porosity",
           "cort_trab_separation", "cort_trab_separation_from_raw", "Step1Params", "TIBIA", "RADIUS", "ipl_ops"]
