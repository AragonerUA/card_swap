"""CLI.

  python -m card_swap detect --photos DIR --out cards.jsonl [--weights yolov8_instance_segmentation.pt]
      Finds the card in every photo with the production YOLO segmentation model (GPU box / SageMaker).
  python -m card_swap from-mrz --photos DIR --labels coco.json --out cards.jsonl
      Finds the card from labelled TD1 MRZ lines (3 lines), no models (laptop).
  python -m card_swap swap --photos DIR --cards cards.jsonl --templates TPL_DIR --out OUT_DIR
      Replaces the found cards with template cards; writes images, annotations.jsonl and previews.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_photos(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def read(path):
    """BGR image; cv2.imread applies the EXIF orientation, so detection and swap see the same pixels."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def mask_to_poly(mask):
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()


def poly_to_mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    return m


def cmd_detect(a):
    from .detect import ProdCardDetector
    det = ProdCardDetector(a.weights, a.conf)
    root, stat = Path(a.photos), Counter()
    with open(a.out, "w") as out:
        for p in list_photos(root):
            rec = {"photo": str(p.relative_to(root))}
            try:
                r = det(read(p))
            except Exception as e:  # keep going over a bad file
                r, rec["error"] = None, repr(e)[:200]
            if r is None:
                rec.setdefault("error", "card_not_found")
            else:
                rec.update(quad=np.round(r["quad"], 1).tolist(), cls=r["cls"], mask_poly=mask_to_poly(r["mask"]))
            stat["found" if r is not None else rec["error"][:40]] += 1
            out.write(json.dumps(rec) + "\n")
    print(dict(stat))


def cmd_from_mrz(a):
    from .swap import quad_from_mrz
    coco = json.load(open(a.labels))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    want = a.categories.split(",")
    lines = {}
    for an in coco["annotations"]:
        name = cats.get(an["category_id"])
        if name not in want:
            continue
        seg = an["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        lines.setdefault(an["image_id"], {})[name] = np.float32(seg[0]).reshape(-1, 2)
    imgs = [im for im in coco["images"] if not (a.skip_synthetic and im.get("synthetic"))]
    root, stat = Path(a.photos), Counter()
    stat["skipped_not_all_mrz_lines"] = sum(sorted(lines.get(im["id"], {})) != sorted(want) for im in imgs)
    imgs = [im for im in imgs if sorted(lines.get(im["id"], {})) == sorted(want)]  # TD1 photos only
    if a.limit:
        imgs = [imgs[i] for i in sorted(np.random.default_rng(a.seed).permutation(len(imgs))[:a.limit])]
    with open(a.out, "w") as out:
        for im in imgs:
            rec = {"photo": im["file_name"]}
            ls = lines[im["id"]]
            try:
                photo = read(root / im["file_name"])
            except ValueError:
                photo, rec["error"] = None, "cannot_read"
            if photo is not None and photo.shape[:2] != (im["height"], im["width"]):
                rec["error"] = "size_mismatch"  # EXIF-rotated: labels are in other pixel coordinates
            elif photo is not None:
                polys = [ls[k] for k in want]
                r = quad_from_mrz(photo, polys, None)
                if r is None or not r[1]:
                    rec["error"] = "card_edges_not_found"
                else:
                    rec.update(quad=np.round(r[0], 1).tolist(), cover_polys=[np.round(p, 1).tolist() for p in polys])
            stat["found" if "quad" in rec else rec["error"]] += 1
            out.write(json.dumps(rec) + "\n")
    print(dict(stat))


def draw(img, quad, fields):
    vis = img.copy()
    t = max(2, img.shape[1] // 500)
    cv2.polylines(vis, [np.asarray(quad, np.int32)], True, (0, 200, 255), t)
    cv2.circle(vis, tuple(int(v) for v in quad[0]), 3 * t, (0, 200, 255), -1)  # template's top-left corner
    for i, p in enumerate(fields.values()):
        cv2.polylines(vis, [np.asarray(p, np.int32)], True, [(0, 0, 255), (0, 180, 0), (255, 0, 0)][i % 3], t)
    return vis


def cmd_swap(a):
    from .detect import regular
    from .swap import load_templates, swap_card
    tpls = load_templates(a.templates)
    names = sorted(tpls)
    if a.only:
        names = [n for n in names if any(s in n for s in a.only.split(","))]
    if not names:
        sys.exit("no templates")
    rng = np.random.default_rng(a.seed)
    root, out = Path(a.photos), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if a.vis:
        (out / "vis").mkdir(exist_ok=True)
    stat = Counter()
    recs = [json.loads(line) for line in open(a.cards)]
    if a.limit:
        recs = [recs[i] for i in sorted(rng.permutation(len(recs))[:a.limit])]
    with open(out / "annotations.jsonl", "w") as ann:
        for rec in recs:
            if "quad" not in rec:
                stat["no_card"] += 1
                continue
            quad = np.asarray(rec["quad"], np.float32)
            if not regular(quad):
                stat["card_not_id1_shape"] += 1
                continue
            photo = read(root / rec["photo"])
            om = poly_to_mask(rec["mask_poly"], photo.shape) if rec.get("mask_poly") else None
            for k in range(a.per_photo):
                tpl = tpls[names[int(rng.integers(len(names)))]]
                r = swap_card(photo, quad, tpl, rng, light=a.light, orig_mask=om, min_cover=a.min_cover,
                              rotate180=bool(rng.random() < a.rotate180_p), must_cover=rec.get("cover_polys"))
                if r is None:
                    stat["rejected"] += 1
                    continue
                stem = f"{Path(rec['photo']).with_suffix('').as_posix().replace('/', '__')}__{k}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), r["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                ann.write(json.dumps({"image": f"images/{stem}.jpg", "photo": rec["photo"], "template": tpl.name,
                                      "size": [photo.shape[1], photo.shape[0]],
                                      "quad": np.round(r["quad"], 1).tolist(),
                                      "fields": {n: np.round(p, 1).tolist() for n, p in r["fields"].items()},
                                      "meta": tpl.meta}, ensure_ascii=False) + "\n")
                if a.vis:
                    v = draw(r["image"], r["quad"], r["fields"])
                    s = 1000 / max(v.shape[:2])
                    cv2.imwrite(str(out / "vis" / f"{stem}.jpg"), cv2.resize(v, None, fx=min(1, s), fy=min(1, s)))
                stat["ok"] += 1
    print(dict(stat))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="card_swap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="find cards with the production YOLO model")
    d.add_argument("--photos", required=True)
    d.add_argument("--out", required=True, help="cards.jsonl")
    d.add_argument("--weights", default=None, help="default: $APP_VERACITY_SEGMENTATION_MODEL_PATH")
    d.add_argument("--conf", type=float, default=None)
    m = sub.add_parser("from-mrz", help="find cards from labelled TD1 MRZ lines (no models)")
    m.add_argument("--photos", required=True, help="root the COCO file_name paths are relative to")
    m.add_argument("--labels", required=True, help="COCO json with the MRZ line polygons")
    m.add_argument("--out", required=True, help="cards.jsonl")
    m.add_argument("--categories", default="mrz_line1,mrz_line2,mrz_line3", help="the 3 MRZ line categories, top to bottom")
    m.add_argument("--skip-synthetic", action="store_true", help="drop images marked synthetic in the COCO file")
    m.add_argument("--limit", type=int, default=0, help="random subset of this many images (0 = all)")
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("swap", help="replace found cards with template cards")
    s.add_argument("--photos", required=True)
    s.add_argument("--cards", required=True, help="cards.jsonl from `detect` (or hand-made: photo + quad)")
    s.add_argument("--templates", required=True, help="folder with templates.json and card images")
    s.add_argument("--out", required=True)
    s.add_argument("--only", default="", help="comma-separated substrings of template names to use")
    s.add_argument("--per-photo", type=int, default=1)
    s.add_argument("--limit", type=int, default=0, help="random subset of this many photos (0 = all)")
    s.add_argument("--light", choices=["match", "simple"], default="match")
    s.add_argument("--min-cover", type=float, default=0.97)
    s.add_argument("--rotate180-p", type=float, default=0.0, help="share of cards put upside down")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--vis", action="store_true", help="also write previews with the card and field polygons")
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "from-mrz": cmd_from_mrz, "swap": cmd_swap}[a.cmd](a)


if __name__ == "__main__":
    main()
