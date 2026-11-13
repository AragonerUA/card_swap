"""Replace the real ID-1 card in a photo with a template card: homography template corners -> card corners
in the photo, light transferred from the real card, template field polygons mapped by the same homography."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import grabcut_quad, snap_quad_to_edges
from .relight import LIGHT


@dataclass
class Template:
    """Flat card image (the card fills the image) and labelled polygons in its pixel coordinates."""
    name: str
    image: np.ndarray
    fields: dict = field(default_factory=dict)      # label -> (N, 2) float32
    meta: dict = field(default_factory=dict)        # passed through to the output annotation
    corner_radius: float = 0.04                     # of the card width; ID-1 is 3.18 / 85.6 = 0.037


def load_templates(folder):
    """`folder/templates.json`: {name: {"image": "<file>" (default <name>.png), "fields": {label: [[x, y], ...]},
    "corner_radius": 0.04, ...any other keys -> meta}}. The older Swiss format {"lines": [q1, q2, q3]} is read
    as fields mrz_line1..3."""
    folder = Path(folder)
    out = {}
    for name, m in json.load(open(folder / "templates.json")).items():
        m = dict(m)
        img = cv2.imread(str(folder / m.pop("image", f"{name}.png")))
        if img is None:
            raise FileNotFoundError(f"template image for {name} not found in {folder}")
        fields = m.pop("fields", {})
        for i, q in enumerate(m.pop("lines", []), 1):
            fields.setdefault(f"mrz_line{i}", q)
        fields = {k: np.asarray(v, np.float32).reshape(-1, 2) for k, v in fields.items()}
        out[name] = Template(name, img, fields, m, float(m.pop("corner_radius", 0.04)))
    return out


def card_mask(h, w, radius):
    """Card mask with rounded corners in template coordinates."""
    m = np.zeros((h, w), np.uint8)
    r = max(1, int(radius * w))
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def fit_orientation(quad, tpl_hw, rotate180=False):
    """Roll the TL,TR,BR,BL corners so the template's long side lands on the card's long side."""
    q = np.asarray(quad, np.float32)
    top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    side = np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])
    th, tw = tpl_hw
    if (tw >= th) != (top >= side):
        q = np.roll(q, -1, axis=0)
    return np.roll(q, 2, axis=0) if rotate180 else q


def swap_card(target, quad, tpl, rng, light="match", orig_mask=None, min_cover=0.97, rotate180=False,
              require_fields_inside=True, must_cover=None, must_cover_min=0.995):
    """Put template `tpl` onto the card at `quad` (TL,TR,BR,BL in the photo).

    orig_mask: mask of the real card (e.g. from the prod detector); the new card must cover at least
    `min_cover` of it, otherwise pieces of the old document would stay visible.
    must_cover: polygons on the old card (e.g. its labelled MRZ lines) that the new card must cover by at least
    `must_cover_min`, otherwise the old text would show next to the new card.
    Returns dict(image, quad, fields, H), or None if the result is rejected.
    """
    th, tw = tpl.image.shape[:2]
    Ht, Wt = target.shape[:2]
    q = fit_orientation(quad, (th, tw), rotate180)
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    H = cv2.getPerspectiveTransform(corners, q.astype(np.float32)).astype(np.float64)
    img_t, m0 = tpl.image, card_mask(th, tw, tpl.corner_radius)
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    f = 1.0
    if scale < 0.8:  # pre-shrink the template so the warp does not alias
        f = scale / 0.9
        size = (max(1, int(tw * f)), max(1, int(th * f)))
        img_t = cv2.resize(img_t, size, interpolation=cv2.INTER_AREA)
        m0 = cv2.resize(m0, size, interpolation=cv2.INTER_NEAREST)
    Hw = H @ np.diag([1 / f, 1 / f, 1])
    warped = cv2.warpPerspective(img_t, Hw, (Wt, Ht), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(m0, Hw, (Wt, Ht), flags=cv2.INTER_NEAREST)
    if mask.sum() < 255 * 1000 or (mask > 0).mean() > 0.95:
        return None
    if must_cover:
        old = np.zeros_like(mask)
        for poly in must_cover:
            cv2.fillPoly(old, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
        if old.any() and (mask[old > 0] > 0).mean() < must_cover_min:
            return None
    om = None
    if orig_mask is not None:
        om = np.asarray(orig_mask) > 0
        if not om.any() or (mask[om] > 0).mean() < min_cover:
            return None
    img = LIGHT[light](target, warped, mask, rng, orig_mask=om)
    fields = {k: cv2.perspectiveTransform(v[None], H)[0] for k, v in tpl.fields.items()}
    if require_fields_inside:
        for p in fields.values():
            if p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max() >= Wt or p[:, 1].max() >= Ht:
                return None
    return {"image": img, "quad": q, "fields": fields, "H": H}


# --- older mode: no card detector, the card is located from its labelled MRZ lines (Swiss licences) ---

def canon_lines(lines):
    """MRZ line quads (any corner order) -> corners TL,TR,BR,BL in the document frame (line 1 above the last)."""
    lines = [np.asarray(q, np.float32) for q in lines]
    longest = max(lines, key=lambda q: max(cv2.minAreaRect(q)[1]))
    axis = np.linalg.svd(longest - longest.mean(0))[2][0].astype(np.float32)  # reading axis (sign unknown)
    down = np.array([-axis[1], axis[0]], np.float32)
    if (lines[-1].mean(0) - lines[0].mean(0)) @ down < 0:
        down = -down
    right = np.array([down[1], -down[0]], np.float32)
    out = []
    for q in lines:
        rect = cv2.boxPoints(cv2.minAreaRect(q)).astype(np.float32)
        rel = rect - rect.mean(0)
        idx = {(bool(a > 0), bool(b > 0)): k for k, (a, b) in enumerate(zip(rel @ right, rel @ down))}
        if len(idx) != 4:
            return None
        out.append(rect[[idx[(False, False)], idx[(True, False)], idx[(True, True)], idx[(False, True)]]])
    return out


# ID-1 card (mm) and where its 3-line TD1 MRZ sits on it: measured on real photos, matches ICAO 9303
# (line pitch 4.23 mm); vertical spread < 1 mm, horizontal ~1.5 mm (looseness of the labelled boxes).
ID1_MM = (85.6, 53.98)
TD1_MRZ_MM = {"left": 3.4, "right": 82.1, "centres": (38.4, 42.6, 47.0), "height": 4.3}


def td1_prior_lines():
    l, r, h = TD1_MRZ_MM["left"], TD1_MRZ_MM["right"], TD1_MRZ_MM["height"] / 2
    return [np.float32([[l, y - h], [r, y - h], [r, y + h], [l, y + h]]) for y in TD1_MRZ_MM["centres"]]


def quad_from_mrz(target, tgt_lines, tpl, use=slice(None), bands=((-0.10, 0.10), (-0.04, 0.04), (-0.04, 0.04),
                                                                   (-0.04, 0.04)), max_corner_shift=0.15,
                  max_aspect_dev=0.08):
    """Card quad in the photo predicted from labelled MRZ lines, then snapped to the card's straight edges.

    3 lines (TD1): the card is placed by the standard TD1 geometry (`TD1_MRZ_MM`), so each edge is known to a
    few mm and is searched in a narrow `bands` window (top, right, bottom, left; the top is farthest from the
    MRZ). Other line counts: placed by the template's own MRZ (`use` picks the lines) and refined by GrabCut.
    Returns (quad, refined: bool) or None."""
    gl = canon_lines(tgt_lines)
    if gl is None:
        return None
    if len(gl) == 3:
        A, _ = cv2.estimateAffine2D(np.concatenate(td1_prior_lines()), np.concatenate(gl), method=cv2.LMEDS)
        if A is None:
            return None
        w, h = ID1_MM
        approx = cv2.transform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], A)[0]
        s = snap_quad_to_edges(target, approx, band=bands, tilt=4.0)
        short = np.linalg.norm(approx[3] - approx[0])
        if s is None or np.linalg.norm(s - approx, axis=1).max() > max_corner_shift * short:
            return approx, False  # no edges, or they drifted too far from where the MRZ puts the card
        sd = [np.linalg.norm(s[(i + 1) % 4] - s[i]) for i in range(4)]
        if abs((sd[0] + sd[2]) / (sd[1] + sd[3]) / (w / h) - 1) > max_aspect_dev:
            return approx, False  # not ID-1 shaped: the MRZ is not where TD1 puts it on this card
        return s, True
    names = sorted(k for k in tpl.fields if k.startswith("mrz_line"))
    tl = canon_lines([tpl.fields[k] for k in names])
    if tl is None or len(tl) != len(gl):
        return None
    A, _ = cv2.estimateAffine2D(np.concatenate(tl[use]), np.concatenate(gl[use]), method=cv2.LMEDS)
    if A is None:
        return None
    th, tw = tpl.image.shape[:2]
    approx = cv2.transform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])[None], A)[0]
    q = grabcut_quad(target, approx)
    base = approx if q is None else q
    s = snap_quad_to_edges(target, base)  # sides onto the card's real straight edges
    if s is not None and np.linalg.norm(s - approx, axis=1).max() <= 0.3 * np.linalg.norm(approx[1] - approx[0]):
        return s, True
    return base, q is not None
