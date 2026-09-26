"""Stage IPL's AUTOMATIC evaluation run of each of the 63 OS_LH measurements whose delivered evaluation was later
corrected by an operator, as a normal, single-version measurement folder the existing harness can read unchanged.

Source   <IPLDT_LAB_ROOT>/PFJOA/XCT_full_grab/edited_all/<Tag>/        (VMS versions of the scanner fetch, read-only)
Mapping  AUTOMATIC_VERSION_SET.json of the internal decode (IPLDT_AUTOMATIC_VERSION_SET; not distributed)
Target   <IPLDT_LAB_ROOT>/Cross_validation_IPL/OS_LH_AUTO/<Group>/<Sub>/<meas>/
The inputs are not public; the script documents how the validation set was assembled.

Products are HARDLINKED (same NTFS volume) so the 35 GB is not duplicated; only the two rendered
contour rasters are new bytes.

The two _CT rasters must be GENERATED, not copied: the shipped <base>_CORT_MASK_CT.AIM is the REDO's
cortical contour and differs from the automatic one by up to 233,908 voxels (Distal/CKD/345857).  The periosteal
<base>_CT.AIM *is* copied -- every one of the 129 runs read the same periosteal GOBJ version.

Run:  python validation/stage_automatic_set.py [--limit N] [--force]
"""
import argparse, json, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datapaths import lab_path  # noqa: E402  (non-public data roots: validation/datapaths.py)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ipldt.io import read_aim, write_aim
from ipldt.contour.gobj_file import read_gobj, path as gobj_path
from ipldt.contour.render import polygon_fill

SRC = lab_path("PFJOA/XCT_full_grab/edited_all")
DST = lab_path("Cross_validation_IPL/OS_LH_AUTO")
MAP = os.environ.get("IPLDT_AUTOMATIC_VERSION_SET") or os.path.join(SRC, "AUTOMATIC_VERSION_SET.json")   # internal decode
VARIANT = 8          # the chain-path variant validated in syn_render (0 mismatches vs IPL's own export, 63/63)

# fetched name (unversioned)      ->  name the oslh layout looks for
COPY = [
    ("{b}.AIM",            "{b}.AIM"),                        # greyscale
    ("{b}_CT.AIM",         "{b}_CT.AIM"),                     # IPL's periosteal rendering (shared by both runs)
    ("{b}_SEG.AIM",        "{b}_SEG.AIM"),
    ("{b}_TRAB_TH.AIM",    "{b}_TRAB_TH_COMPRESSED.AIM"),     # layout wants _DECOMPRESSED or _COMPRESSED
    ("{b}_TRAB_SP.AIM",    "{b}_TRAB_SP_COMPRESSED.AIM"),
    ("{b}_TRAB_1N.AIM",    "{b}_TRAB_1N_COMPRESSED.AIM"),
    ("{b}_CORT_TH.AIM",    "{b}_CORT_TH_COMPRESSED.AIM"),
    ("{b}_TRAB_SEG.AIM",   "{b}_TRAB_SEG.AIM"),               # IPL's own, now available -- not read by the
    ("{b}_CORT_SEG.AIM",   "{b}_CORT_SEG.AIM"),               # default layout, staged for the stricter run
    ("{b}_TRAB_MASK.AIM",  "{b}_TRAB_MASK.AIM"),              # IPL's own raster: STEP 1 ground truth
    ("{b}_CORT_MASK.AIM",  "{b}_CORT_MASK.AIM"),
    ("{b}_PORE.AIM",       "{b}_PORE.AIM"),                   # the Burghardt pore cascade's product (Ct.Po)
    ("{b}_CORT_SP.AIM",    "{b}_CORT_SP_COMPRESSED.AIM"),     # dt_spacing of the inverted pore map (Ct.Po.Dm)
]
RENDER = [("{b}_TRAB_MASK.GOBJ", "{b}_TRAB_MASK_CT.AIM"),
          ("{b}_CORT_MASK.GOBJ", "{b}_CORT_MASK_CT.AIM")]


def render_gobj(path, dim, pos):
    """Even-odd parity fill over all chains of each slice at once, chain pixels forced on."""
    _hdr, slices = read_gobj(path)
    G = np.zeros(dim[::-1], bool)
    for s in slices:
        z = s["z"] - pos[2]
        if not (0 <= z < dim[2]) or not s["contours"]:
            continue
        inside = np.zeros((dim[1], dim[0]), bool)
        chain = np.zeros((dim[1], dim[0]), bool)
        for c in s["contours"]:
            x, y = gobj_path(c["codes"], (c["h"][1] + c["h"][13], c["h"][2] + c["h"][14]), VARIANT)
            xx, yy = x - pos[0], y - pos[1]
            if not ((xx >= 0) & (xx < dim[0]) & (yy >= 0) & (yy < dim[1])).all():
                xx, yy = np.clip(xx, 0, dim[0] - 1), np.clip(yy, 0, dim[1] - 1)
            ch, ins = polygon_fill(list(zip(xx, yy)), (dim[1], dim[0]))
            inside ^= ins
            chain |= ch
        G[z] = inside | chain
    return G


def link(src, dst, force):
    if os.path.exists(dst):
        if not force:
            return "kept"
        os.remove(dst)
    try:
        os.link(src, dst)                      # NTFS hardlink: no bytes copied
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dst", default=DST)
    a = ap.parse_args()

    mapping = json.load(open(MAP, encoding="utf-8"))
    tags = sorted(mapping)[: a.limit or None]
    t0 = time.time()
    stats = {"linked": 0, "copied": 0, "kept": 0, "rendered": 0}
    problems, staged = [], []

    for i, tag in enumerate(tags, 1):
        e = mapping[tag]
        b = e["base"]
        group, sub, meas = tag.split("_", 2)          # Distal_CKD_345857 -> Distal / CKD / 345857
        src = os.path.join(SRC, tag)
        dst = os.path.join(a.dst, group, sub, meas)
        os.makedirs(dst, exist_ok=True)
        files = e["files"]

        for pat, target in COPY:
            name = pat.format(b=b)
            v = files.get(name)
            if v is None:
                if name.endswith(("_TRAB_SEG.AIM", "_CORT_SEG.AIM", "_TRAB_MASK.AIM", "_CORT_MASK.AIM")):
                    v = e["automatic_run"]            # present in the fetch, just not listed in `files`
                elif name.endswith("_CT.AIM"):
                    v = 1                             # the periosteal rendering has a single version
                else:
                    problems.append(f"{tag}: no mapping for {name}")
                    continue
            s = os.path.join(src, f"{name};{v}")
            if not os.path.exists(s):
                problems.append(f"{tag}: missing {name};{v}")
                continue
            stats[link(s, os.path.join(dst, target.format(b=b)), a.force)] += 1

        # the two rasters that must be generated from the AUTOMATIC contours
        ctp = os.path.join(dst, f"{b}_CT.AIM")
        if not os.path.exists(ctp):
            problems.append(f"{tag}: no periosteal {b}_CT.AIM, cannot render")
            continue
        per = read_aim(ctp)
        dim, pos = tuple(per["dim"]), tuple(per["pos"])
        for gpat, opat in RENDER:
            gname = gpat.format(b=b)
            v = files.get(gname, e["automatic_run"])
            g = os.path.join(src, f"{gname};{v}")
            out = os.path.join(dst, opat.format(b=b))
            if os.path.exists(out) and not a.force:
                continue
            if not os.path.exists(g):
                problems.append(f"{tag}: missing {gname};{v}")
                continue
            G = render_gobj(g, dim, pos)
            write_aim(out, G.astype(np.uint8) * 127, per["header"])
            stats["rendered"] += 1

        staged.append(dict(tag=tag, id=f"{group}/{sub}/{meas}", base=b, cls=e["cls"], preset=e["preset"],
                           automatic_run=e["automatic_run"], variants=e["variants"], dst=dst))
        if i % 10 == 0 or i == len(tags):
            print(f"[{time.time()-t0:6.1f}s] {i}/{len(tags)}  {tag}", flush=True)

    out = os.path.join(a.dst, "staged_manifest.json")
    json.dump(staged, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nstaged {len(staged)} measurements -> {a.dst}")
    print(f"  files: {stats['linked']} hardlinked, {stats['copied']} copied, {stats['kept']} already there, "
          f"{stats['rendered']} contour rasters rendered")
    print(f"  manifest: {out}")
    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print("   ", p)
    else:
        print("  no problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
