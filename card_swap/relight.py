"""Put a flat template card into a photo so it looks shot in that scene: the light, colour cast, highlights,
sharpness and noise are taken from the real card that is being replaced."""
import cv2
import numpy as np


def _gray(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _lowfreq(img, sigma, f):
    """Gaussian blur with a large sigma, computed on a copy downscaled by f (fast) and upscaled back."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma * f))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(out, rng):
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(60, 92))])[1], 1)


def light_simple(target, warped, mask, rng, orig_mask=None):
    """v1: shading, colour cast and highlights of the original card; random blur and noise."""
    t = target.astype(np.float32)
    ys, xs = np.nonzero(mask)
    h = max(int(ys.max() - ys.min()), 10)
    f = min(1.0, 60.0 / h)  # low-frequency maps at ~60 px card height
    k = max(3, h // 25) | 1
    clean = cv2.morphologyEx(target, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))  # text removed
    g_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = mask > 0
    shade = _lowfreq(g_clean, h / 8, f)
    shade = np.clip(shade / (np.percentile(shade[m], 70) + 1e-3), 0.45, 1.15)
    col = _lowfreq(clean.astype(np.float32), h / 4, f)
    tint = np.clip(0.7 + 0.3 * col[m].mean(0) / (col[m].mean() + 1e-3), 0.85, 1.15)
    w = warped.astype(np.float32) * shade[..., None] * tint
    spec = np.clip(g_clean - _lowfreq(g_clean, h / 10, min(1.0, 2 * f)) - 18, 0, 255)  # real bright spots
    w = w + spec[..., None] * 0.8
    sigma = rng.uniform(0.0, 1.0) * h / 350
    if sigma > 0.3:
        w = cv2.GaussianBlur(w, (0, 0), sigma)
    w += rng.normal(0, rng.uniform(1.0, 4.0), w.shape).astype(np.float32)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.2)[..., None]
    return _jpeg(np.clip(w * a + t * (1 - a), 0, 255).astype(np.uint8), rng)


def light_match(target, warped, mask, rng, orig_mask=None, glare_p=0.0, shadow_p=0.0):
    """v3 (default): photometric match of the warped card to the real card it replaces (`orig_mask`, else
    the new card's area): spatial shading, tone curve (lightness percentiles), colour cast and saturation,
    real highlights, matched blur and noise. Optional synthetic glare / shadow (off by default)."""
    t = target.astype(np.float32)
    m = mask > 0
    om = m if orig_mask is None else (np.asarray(orig_mask) > 0) & m
    if om.sum() < 0.5 * m.sum():
        om = m
    ys, xs = np.nonzero(m)
    h = max(int(ys.max() - ys.min()), 10)
    f = min(1.0, 60.0 / h)
    k = max(3, h // 25) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean_t = cv2.morphologyEx(target, cv2.MORPH_CLOSE, se).astype(np.float32)
    # 1. shading across the card relative to its median (shadow of a hand, vignetting, light falloff)
    Lt = _lowfreq(_gray(clean_t), h / 8, f)
    shade = np.clip(Lt / (np.median(Lt[om]) + 1e-3), 0.35, 1.3)
    # 2. tone curve: histogram matching of lightness, new card -> real card with shading removed
    #    (mid-rank, so a flat template colour maps to the middle of its rank range, not to its edge)
    lab_t = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_w = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    hist = np.bincount(lab_w[..., 0][m].astype(np.int64), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / hist.sum()
    lut = np.percentile((lab_t[..., 0] / shade)[om], 100 * (cdf - hist / hist.sum() / 2))
    L = lut[lab_w[..., 0].astype(np.int64)] * shade
    # 3. colour: chroma spread and cast of the real card (faded / tinted photos)
    ab = []
    for c in (1, 2):
        wc, tc = lab_w[..., c] - 128, lab_t[..., c] - 128
        s = np.clip((tc[om].std() + 1) / (wc[m].std() + 1), 0.6, 1.2)
        # the template keeps its own colours; half of the difference to the real card is the scene's cast
        ab.append(128 + wc * s + 0.5 * (tc[om].mean() - wc[m].mean() * s))
    w = cv2.cvtColor(np.clip(np.dstack([L] + ab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # 4. real highlights of the original card
    g_clean = _gray(clean_t)
    spec = np.clip(g_clean - _lowfreq(g_clean, h / 10, min(1.0, 2 * f)) - 18, 0, 255)
    w = w + spec[..., None] * 0.8
    # 5. optional synthetic glare / shadow
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float32)
    cy, cx = ys.mean(), xs.mean()
    if rng.random() < glare_p:  # soft glare blob on the laminate
        px, py = cx + rng.uniform(-.5, .5) * h * 1.5, cy + rng.uniform(-.5, .5) * h
        d = np.hypot((xx - px) / rng.uniform(0.6, 1.6), yy - py) / (h * rng.uniform(0.15, 0.4))
        gl = np.exp(-d ** 2) * rng.uniform(40, 110) * m
        w = w + gl[..., None] * (1 - w / 320)
    if rng.random() < shadow_p:  # soft shadow (hand / phone) over part of the card
        ang = rng.uniform(0, 2 * np.pi); off = rng.uniform(-0.3, 0.4) * h
        d = ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang) - off) / (h * rng.uniform(0.05, 0.3))
        w = w * (1 - rng.uniform(0.2, 0.45) / (1 + np.exp(-d)))[..., None]
    # 6. sharpness: blur the card until its Laplacian energy matches the real card
    gt = _gray(t)
    lt = cv2.Laplacian(gt, cv2.CV_32F)[om].var()
    best, sig = None, 0.0
    for s_ in (0.0, 0.5, 0.8, 1.1, 1.5, 2.0, 2.7, 3.5):
        ww = cv2.GaussianBlur(w, (0, 0), s_) if s_ else w
        e = abs(np.log(cv2.Laplacian(_gray(ww), cv2.CV_32F)[m].var() + 1) - np.log(lt + 1))
        if best is None or e < best:
            best, sig = e, s_
    sig *= rng.uniform(0.85, 1.1)
    if sig > 0.3:
        w = cv2.GaussianBlur(w, (0, 0), sig)
    # 7. noise level of the photo
    hp = gt - cv2.GaussianBlur(gt, (0, 0), 1.0)
    nz = float(np.clip(1.4826 * np.median(np.abs(hp[om])), 0.8, 8))
    w += rng.normal(0, nz, w.shape).astype(np.float32)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.2)[..., None]
    return _jpeg(np.clip(w * a + t * (1 - a), 0, 255).astype(np.uint8), rng)


LIGHT = {"simple": light_simple, "match": light_match}
