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


def light_simple(target, warped, mask, rng):
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


def light_match(target, warped, mask, rng, glare_p=0.35, shadow_p=0.3):
    """v2 (default): photometric match of the warped card to the photo: exposure from the bright parts of
    the original card, spatial shading, colour cast, lifted black level, matched blur and noise; plus
    random glare / shadow."""
    t = target.astype(np.float32)
    m = mask > 0
    ys, xs = np.nonzero(m)
    h = max(int(ys.max() - ys.min()), 10)
    f = min(1.0, 60.0 / h)
    k = max(3, h // 25) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean_t = cv2.morphologyEx(target, cv2.MORPH_CLOSE, se).astype(np.float32)
    clean_w = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, se).astype(np.float32)
    Lt, Lw = _lowfreq(_gray(clean_t), h / 8, f), _lowfreq(_gray(clean_w), h / 8, f)
    # 1. exposure: bright (paper-like) parts of the original card vs of the new card
    gain = np.clip(np.percentile(Lt[m], 95) / (np.percentile(Lw[m], 95) + 1e-3), 0.45, 1.1)
    # 2. shading across the card relative to its median (shadow of a hand, vignetting, light falloff)
    shade = np.clip(Lt / (np.median(Lt[m]) + 1e-3), 0.35, 1.3)
    shade = 1 + (shade - 1) * rng.uniform(0.8, 1.2)
    # 3. colour cast of the scene, partially
    col_t, col_w = clean_t[m].mean(0), clean_w[m].mean(0)
    cast_t, cast_w = col_t / (col_t.mean() + 1e-3), col_w / (col_w.mean() + 1e-3)
    tint = np.clip(1 + (cast_t / cast_w - 1) * rng.uniform(0.3, 0.6), 0.85, 1.18)
    w = warped.astype(np.float32) * gain * shade[..., None] * tint
    # 4. black level / contrast: ink in photos is never pure black
    gt, gw = _gray(t), _gray(w)
    lift = np.clip(np.percentile(gt[m], 1.5) - np.percentile(gw[m], 1.5), 0, 80) * rng.uniform(0.7, 1.0)
    w = lift + w * (1 - lift / 255)
    # 5. real highlights of the original card
    g_clean = _gray(clean_t)
    spec = np.clip(g_clean - _lowfreq(g_clean, h / 10, min(1.0, 2 * f)) - 18, 0, 255)
    w = w + spec[..., None] * 0.8
    # 6. extra glare / shadow in some images
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float32)
    cy, cx = ys.mean(), xs.mean()
    if rng.random() < glare_p:  # glare streak or blob on the laminate
        px, py = cx + rng.uniform(-.5, .5) * h * 1.5, cy + rng.uniform(-.5, .5) * h
        if rng.random() < 0.5:
            ang = rng.uniform(0, np.pi)
            d = np.abs((xx - px) * np.sin(ang) - (yy - py) * np.cos(ang)) / (h * rng.uniform(0.04, 0.2))
        else:
            d = np.hypot((xx - px) / rng.uniform(0.6, 1.6), yy - py) / (h * rng.uniform(0.1, 0.35))
        gl = np.exp(-d ** 2) * rng.uniform(80, 200) * m
        w = w + gl[..., None] * (1 - w / 320)
    if rng.random() < shadow_p:  # soft shadow (hand / phone) over part of the card
        ang = rng.uniform(0, 2 * np.pi); off = rng.uniform(-0.3, 0.4) * h
        d = ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang) - off) / (h * rng.uniform(0.05, 0.3))
        w = w * (1 - rng.uniform(0.25, 0.55) / (1 + np.exp(-d)))[..., None]
    # 7. sharpness: blur the card until its Laplacian energy matches the photo's card
    lt = cv2.Laplacian(gt, cv2.CV_32F)[m].var()
    best, sig = None, 0.0
    for s_ in (0.0, 0.5, 0.8, 1.1, 1.5, 2.0, 2.7, 3.5):
        ww = cv2.GaussianBlur(w, (0, 0), s_) if s_ else w
        e = abs(np.log(cv2.Laplacian(_gray(ww), cv2.CV_32F)[m].var() + 1) - np.log(lt + 1))
        if best is None or e < best:
            best, sig = e, s_
    sig *= rng.uniform(0.8, 1.15)
    if sig > 0.3:
        w = cv2.GaussianBlur(w, (0, 0), sig)
    # 8. noise level of the photo
    hp = gt - cv2.GaussianBlur(gt, (0, 0), 1.0)
    nz = float(np.clip(1.4826 * np.median(np.abs(hp[m])), 0.8, 8))
    w += rng.normal(0, nz, w.shape).astype(np.float32)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.2)[..., None]
    return _jpeg(np.clip(w * a + t * (1 - a), 0, 255).astype(np.uint8), rng)


LIGHT = {"simple": light_simple, "match": light_match}
