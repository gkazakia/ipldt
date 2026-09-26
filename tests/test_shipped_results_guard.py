"""The shipped-directory guard of validation/validate_from_ipl_contour.py (guard_shipped_results).

WHAT IS PINNED HERE
  * a run whose --results is one of the published result directories under validation/results/ (the records the
    paper's numbers are built from), or lies under one, is refused with exit status 2 before anything is written,
    whatever its padding offset or dtype, unless allow_overwrite, in which case a warning naming the offset and the
    dtype is logged and the directory hit is returned;
  * a run into any other directory passes (returns None, logs nothing);
  * the dtype parameter defaults to the engine's LH_DTYPE;
  * validate_dataset.py calls the same function with its own --lh-dtype.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION = os.path.join(REPO, "validation")
for _p in (REPO, VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

vfc = pytest.importorskip("validate_from_ipl_contour", reason="validation/validate_from_ipl_contour.py needs the validation extras")
from ipldt import ormir  # noqa: E402


@pytest.fixture
def dirs(tmp_path):
    shipped = str(tmp_path / "shipped")
    other = str(tmp_path / "elsewhere")
    os.makedirs(shipped)
    os.makedirs(other)
    return shipped, other


def _guard(results, pad, dtype, allow, shipped, logged):
    return vfc.guard_shipped_results(results, pad, allow, shipped_dirs=(shipped,), log_fn=logged.append, lh_dtype=dtype)


def test_shipped_floor_float64_is_refused(dirs, capsys):
    shipped, _ = dirs
    logged = []
    with pytest.raises(SystemExit) as e:
        _guard(shipped, "floor", "float64", False, shipped, logged)
    assert e.value.code == 2
    assert logged == []
    err = capsys.readouterr().err
    assert "REFUSED" in err and "float64" in err and "float32" in err


def test_shipped_floor_float32_is_refused_too(dirs):
    shipped, _ = dirs
    logged = []
    with pytest.raises(SystemExit) as e:
        _guard(shipped, "floor", "float32", False, shipped, logged)
    assert e.value.code == 2
    assert logged == []


def test_non_shipped_ceil_float64_passes(dirs):
    shipped, other = dirs
    logged = []
    assert _guard(other, "ceil", "float64", False, shipped, logged) is None
    assert logged == []


def test_shipped_ceil_float32_is_still_refused_and_subdirectories_count(dirs):
    shipped, _ = dirs
    logged = []
    with pytest.raises(SystemExit) as e:
        _guard(shipped, "ceil", "float32", False, shipped, logged)
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        _guard(os.path.join(shipped, "subset_PFJ-0be66a_R"), "floor", "float64", False, shipped, logged)
    assert logged == []


def test_allow_overwrite_logs_offset_and_dtype(dirs):
    shipped, _ = dirs
    logged = []
    assert _guard(shipped, "floor", "float64", True, shipped, logged) == shipped
    assert len(logged) == 1 and "WARNING" in logged[0] and "'floor'" in logged[0] and "'float64'" in logged[0]


def test_dtype_defaults_to_the_engine_default(dirs):
    shipped, _ = dirs
    assert ormir.LH_DTYPE == "float32"
    logged = []
    assert vfc.guard_shipped_results(shipped, "floor", True, shipped_dirs=(shipped,), log_fn=logged.append) == shipped
    assert len(logged) == 1 and "'float32'" in logged[0]


def test_real_shipped_directories_refuse_float64():
    logged = []
    for d in vfc.SHIPPED_RESULTS_DIRS:
        with pytest.raises(SystemExit) as e:
            vfc.guard_shipped_results(d, "floor", False, log_fn=logged.append, lh_dtype="float64")
        assert e.value.code == 2
    assert logged == []
