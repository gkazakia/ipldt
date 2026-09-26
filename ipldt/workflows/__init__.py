# ipldt.workflows -- command-line entry points in the style of ormir_xct.workflows.
#
# The expected usage is as console scripts (see [project.scripts] in pyproject.toml):
#   ipldt-pipeline         AIM -> IPL-style morphometry (autocontour, compartments, SEG, Tb.Th/Tb.Sp/Tb.N/Ct.Th)
#   ipldt-dt-thickness     IPL /dt_thickness on a binary image (+ optional contour mask)
#   ipldt-dt-spacing       IPL /dt_spacing
#   ipldt-dt-number        IPL /dt_number (1/Tb.N map, Tb.N)
#   ipldt-ipl-morphometry  the dt stage on IPL's own SEG / TRAB_MASK / CORT_MASK AIMs (with a voxel-for-voxel
#                          comparison against IPL's maps) or on any matching NIfTI/MHA images
# They are not meant to be imported.
