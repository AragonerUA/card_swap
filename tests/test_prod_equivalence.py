"""quad_from_mask must give exactly the corners production align_image uses.

Needs a docver-ml-app checkout: DOCVER_ML_APP=/path/to/docver-ml-app pytest tests/test_prod_equivalence.py
(only common/utils/alignment.py and cv_utils.py are loaded, on synthetic masks; scipy is required)."""
import importlib.util
import os
import sys
import types

import cv2
import numpy as np
import pytest

from card_swap.detect import quad_from_mask

REPO = os.environ.get("DOCVER_ML_APP")
pytestmark = pytest.mark.skipif(not REPO, reason="set DOCVER_ML_APP to a docver-ml-app checkout")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prod_corners():
    saved = {k: sys.modules.get(k) for k in ("common", "common.utils", "common.utils.cv_utils", "common.utils.alignment")}
    for p in ("common", "common.utils"):
        sys.modules[p] = types.ModuleType(p)  # skip package __init__ (pulls service deps)
    _load("common.utils.cv_utils", f"{REPO}/common/utils/cv_utils.py")
    al = _load("common.utils.alignment", f"{REPO}/common/utils/alignment.py")
    rec, orig = {}, cv2.getPerspectiveTransform
    al.cv2 = types.SimpleNamespace(**{k: getattr(cv2, k) for k in dir(cv2) if not k.startswith("__")})
    al.cv2.getPerspectiveTransform = lambda s, d: (rec.__setitem__("c", np.array(s)), orig(s, d))[1]

    def corners(mask):
        try:
            al.align_image(np.zeros(mask.shape + (3,), np.uint8), mask)
        except Exception:
            return None
        return rec["c"][[0, 3, 2, 1]]  # prod order tl, bl, br, tr -> TL, TR, BR, BL

    yield corners
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_same_corners_as_prod(prod_corners):
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = np.zeros((600, 700), np.uint8)
        a = np.deg2rad(rng.uniform(-40, 40)); w, h = rng.uniform(300, 560), rng.uniform(200, 360)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T + [350, 300]
        q += rng.normal(0, 8, q.shape)
        cv2.fillConvexPoly(m, q.astype(np.int32), 1)
        if rng.random() < 0.5:  # chop a corner
            c = q[rng.integers(4)].astype(int); s = rng.integers(20, 80)
            cv2.fillConvexPoly(m, np.int32([c + [-s, -s], c + [s, -s], c + [s, s], c + [-s, s]]), 0)
        p, mine = prod_corners(m), quad_from_mask(m)
        assert (p is None) == (mine is None)
        if p is not None:
            np.testing.assert_allclose(mine, p, atol=1e-3)
