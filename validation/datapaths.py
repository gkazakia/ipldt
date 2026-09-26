"""Where the NON-PUBLIC data live: the scans, IPL's products and the IPL probe exports.

Nothing here is needed to install or use ipldt / ORMIR-BQRL, to run the fast tests, or to regenerate the paper's
numbers from the de-identified records under validation/results/.  It is used only by the slow tests, the validation
harness and the figure scripts that draw images, and every one of them skips or stops with a clear message when the
data are absent.

One environment variable sets every default at once:

    IPLDT_LAB_ROOT      a folder with the laboratory's layout, e.g.
                          <IPLDT_LAB_ROOT>/PFJOA/XCT_masks_full_grab/<subject>/        the 21 patellae
                          <IPLDT_LAB_ROOT>/Cross_validation_IPL/OS_LH/<group>/<meas>/  radius / tibia, IPL's delivery
                          <IPLDT_LAB_ROOT>/Cross_validation_IPL/OS_LH_AUTO/...         radius / tibia, first set (automatic runs)
                          <IPLDT_LAB_ROOT>/Cross_validation_IPL/OS_LH_NOEDIT/...       radius / tibia, second set
                          <IPLDT_LAB_ROOT>/Python/scripts/IPL/probes/p15 ... p21/      IPL probe exports
                          <IPLDT_LAB_ROOT>/ipl_probes/p19_open_halves/                 the probe-19 phantom with IPL's
                                                                                       rendering (IPLDT_PROBE19_MIRROR)
                          <IPLDT_LAB_ROOT>/ormir_run_PFJ-0be66a/                       an ORMIR-BQRL run of the Figure 1
                                                                                       scan (IPLDT_FIG1_PRX_MASK)

and the individual variables named where they are used (IPLDT_DATA_ROOT, OSLH_ROOT, IPLDT_PROBE15,
IPLDT_PROBE19_MIRROR, IPLDT_FIG1_PRX_MASK, ...) override single locations.  Unset, lab_path() returns a path under the relative folder 'IPLDT_LAB_ROOT_is_not_set', which
never exists, so a missing-data error names the variable to set.
"""
import os

LAB_ROOT = os.environ.get("IPLDT_LAB_ROOT", "")
MISSING = "IPLDT_LAB_ROOT_is_not_set"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lab_path(rel):
    """<IPLDT_LAB_ROOT>/<rel>, or a never-existing 'IPLDT_LAB_ROOT_is_not_set/<rel>' when the variable is unset."""
    return os.path.join(LAB_ROOT or MISSING, rel).replace("\\", "/")


def have_lab():
    return bool(LAB_ROOT) and os.path.isdir(LAB_ROOT)


def require(path, what):
    """Stop with a clear message when a non-public input is absent."""
    if not os.path.exists(path):
        raise SystemExit(f"{what} is not public and was not found at {path!r}: set IPLDT_LAB_ROOT (or the specific "
                         "variable named in the script) to a folder that holds it.  The published PNG / SVG and the "
                         "*_numbers.json sidecar are the record of this output.")
    return path


def public_path(p):
    """A path as it may be written into a published file: repository-relative, '<IPLDT_LAB_ROOT>/...' for the
    non-public data, never an absolute local path."""
    if isinstance(p, (list, tuple)):
        return [public_path(q) for q in p]
    t = str(p).replace("\\", "/")
    if t.startswith(MISSING + "/"):
        return "<IPLDT_LAB_ROOT>/" + t[len(MISSING) + 1:]
    s = os.path.abspath(str(p)).replace("\\", "/")
    repo = REPO.replace("\\", "/").rstrip("/") + "/"
    if s.lower().startswith(repo.lower()):
        return s[len(repo):]
    if LAB_ROOT:
        lab = os.path.abspath(LAB_ROOT).replace("\\", "/").rstrip("/") + "/"
        if s.lower().startswith(lab.lower()):
            return "<IPLDT_LAB_ROOT>/" + s[len(lab):]
    return "<non-public>/" + os.path.basename(s)
