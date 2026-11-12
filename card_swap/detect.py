"""Where is the card in the photo.

`quad_from_mask` reproduces how production veracity gets document corners from the YOLO segmentation
mask (common/utils/alignment.py: align_image + cv_utils.order_points), but returns the corners instead of
warping. `ProdCardDetector` runs the production YOLOv8 segmentation model (needs ultralytics and the
weights, meant for SageMaker). `grabcut_quad` refines a rough quad (e.g. one predicted from MRZ lines)
when no mask is available.
"""
import os

import cv2
import numpy as np

FACE_CLASS = 5  # YOLO class "face" in the production model, not a document


def _simplify_contour(contour, n_corners=4):
    """Binary search of approxPolyDP epsilon so the contour has exactly n_corners points (prod copy)."""
    lb, ub = 0.0, 1.0
    for _ in range(20):
        k = (lb + ub) / 2.0
        approx = cv2.approxPolyDP(contour, k * cv2.arcLength(contour, True), True)
        if len(approx) > n_corners:
            lb = k
        elif len(approx) < n_corners:
            ub = k
        else:
            return approx
    return contour


def _line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None
    d = (det(line1[0:2], line1[2:4]), det(line2[0:2], line2[2:4]))
    return det(d, xdiff) / div, det(d, ydiff) / div


def _five_to_four(approx):
    """Prod handling of a 5-corner approximation (one corner cut, e.g. by the frame): drop the extra
    point and put the missing corner at the intersection of the adjacent sides."""
    pts = approx[:, 0]
    ix_min = np.argsort(approx, axis=0)[:2][:, 0][:, 0]
    ix_max = np.argsort(approx, axis=0)[-2:][:, 0][:, 0]
    iy_min = np.argsort(approx, axis=0)[:2][:, 0][:, 1]
    iy_max = np.argsort(approx, axis=0)[-2:][:, 0][:, 1]
    extra = np.setdiff1d(np.arange(5), np.concatenate([ix_min, ix_max]))[0]
    ex, ey = pts[extra]
    left = abs(ex - pts[ix_min[0]][0]) < abs(pts[ix_max[1]][0] - ex)
    side = ix_min if left else ix_max
    y_up, y_low = pts[side][:, 1].min(), pts[side][:, 1].max()
    near_up = abs(ey - y_up) < abs(ey - y_low)
    up = near_up if left else not near_up  # prod: on the right side the up/bottom test is inverted
    other = iy_min if up else iy_max
    p = _line_intersection(pts[side].reshape(-1).tolist(), pts[other].reshape(-1).tolist())
    if p is None:
        return None
    new = np.array([[int(p[0]), int(max(p[1], 0))]])
    drop = side[np.argmin(pts[side][:, 1])] if up else side[np.argmax(pts[side][:, 1])]
    return np.append(np.delete(pts, [drop, extra], 0), new, axis=0).astype(np.float32)


def _order_tl_tr_br_bl(points):
    """cv_utils.order_points (tl by x then y, br = farthest from tl) in TL, TR, BR, BL order."""
    x_sorted = points[np.argsort(points[:, 0])]
    left, right = x_sorted[:2], x_sorted[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    br, tr = right[np.argsort(np.linalg.norm(right - tl, axis=1))[::-1]]
    return np.array([tl, tr, br, bl], np.float32)


def quad_from_mask(mask):
    """Card corners (TL, TR, BR, BL; float32 4x2) from a binary card mask, as production does; None if
    the mask does not give 4 corners."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)  # prod: reach the actual edges
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=len))
    approx = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        approx = _simplify_contour(hull)
    if len(approx) == 5:
        corners = _five_to_four(approx)
    elif len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        return None
    if corners is None:
        return None
    return _order_tl_tr_br_bl(corners)


def _scale_masks(masks, shape):
    """Masks (H, W, N) from the letterboxed model resolution to the image resolution (prod scale_image)."""
    h1, w1 = masks.shape[:2]
    if (h1, w1) == tuple(shape[:2]):
        return masks
    gain = min(h1 / shape[0], w1 / shape[1])
    px, py = (w1 - shape[1] * gain) / 2, (h1 - shape[0] * gain) / 2
    top, left = round(py - 0.1), round(px - 0.1)
    bottom, right = round(h1 - py + 0.1), round(w1 - px + 0.1)
    masks = cv2.resize(masks[top:bottom, left:right], (shape[1], shape[0]))
    return masks[:, :, None] if masks.ndim == 2 else masks


class ProdCardDetector:
    """Production YOLOv8 instance segmentation -> largest document mask -> prod corners.

    Weights: `model_path` or APP_VERACITY_SEGMENTATION_MODEL_PATH (as in veracity). Production also runs
    the orientation model before YOLO; here the photo is used as is (EXIF-rotated on load).
    """

    def __init__(self, model_path=None, conf=None):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(model_path or os.environ.get("APP_VERACITY_SEGMENTATION_MODEL_PATH",
                                                       "yolov8_instance_segmentation.pt"))
        self.conf = conf

    def mask(self, image):
        """Largest non-face document mask at image resolution (uint8 0/1) and its class id, or (None, None)."""
        kw = {"verbose": False} if self.conf is None else {"verbose": False, "conf": self.conf}
        r = self.model.predict(image, **kw)[0]
        if r.masks is None or len(r.boxes) == 0:
            return None, None
        cls = r.boxes.cls.cpu().numpy().astype(int)
        keep = np.where(cls != FACE_CLASS)[0]
        if not len(keep):
            return None, None
        masks = r.masks.data.cpu().numpy()[keep].transpose(1, 2, 0)
        masks = _scale_masks(masks, image.shape) > 0.5
        i = int(masks.reshape(-1, masks.shape[2]).sum(0).argmax())
        return masks[:, :, i].astype(np.uint8), int(cls[keep][i])

    def __call__(self, image):
        """dict(quad, mask, cls) or None."""
        m, c = self.mask(image)
        if m is None:
            return None
        q = quad_from_mask(m)
        return None if q is None else {"quad": q, "mask": m, "cls": c}


def grabcut_quad(target, approx, max_shift=0.3):
    """Card corners near `approx` (4x2, TL,TR,BR,BL) via GrabCut; None if not found or inconsistent."""
    Ht, Wt = target.shape[:2]
    wcard = float(np.linalg.norm(approx[1] - approx[0]))
    c = approx.mean(0)
    big = c + (approx - c) * 1.6
    x0, y0 = np.clip(big.min(0), 0, [Wt - 1, Ht - 1]).astype(int)
    x1, y1 = np.clip(big.max(0), 0, [Wt - 1, Ht - 1]).astype(int)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    k = 600.0 / max(x1 - x0, y1 - y0)  # work on a small crop
    crop = cv2.resize(target[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    m = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    for f, lab in ((1.45, cv2.GC_PR_BGD), (0.85, cv2.GC_PR_FGD), (0.5, cv2.GC_FGD)):
        cv2.fillConvexPoly(m, (((c + (approx - c) * f) - [x0, y0]) * k).astype(np.int32), lab)
    try:
        cv2.grabCut(crop, m, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    full = np.zeros((Ht, Wt), np.uint8)
    full[y0:y1, x0:x1] = cv2.resize(fg, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    quad = quad_from_mask(full)
    if quad is None:
        return None
    order = [int(np.argmin(np.linalg.norm(quad - p, axis=1))) for p in approx]  # keep approx's corner order
    if len(set(order)) != 4:
        return None
    quad = quad[order]
    if np.linalg.norm(quad - approx, axis=1).max() > max_shift * wcard:
        return None
    return quad if regular(quad) else None


def regular(q, aspect=(1.3, 1.9)):
    """Near-rectangular ID-1 card under mild perspective (either orientation)."""
    q = np.asarray(q, np.float32)
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    if min(sides) < 1 or not (0.8 < sides[0] / sides[2] < 1.25 and 0.8 < sides[1] / sides[3] < 1.25):
        return False
    a = (sides[0] + sides[2]) / (sides[1] + sides[3])
    if not (aspect[0] < a < aspect[1] or aspect[0] < 1 / a < aspect[1]):
        return False
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        ang = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        if not 70 < ang < 110:
            return False
    return True
