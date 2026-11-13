# card_swap

Synthetic training images from real photos: the real ID-1 card in a photo is replaced with a template card
(any country, any side), and the photo's light is kept. Hands, table, perspective, shadows, glare,
blur, noise and JPEG come from the real photo; the content of the card comes from the template. Labels
drawn on the template (MRZ lines, fields) come out mapped into the photo exactly, through the same
homography.

First used to make Swiss driving licence backs for the MRZ line detector.

## How it works

1. **Find the card**, the way production veracity does: YOLOv8 instance segmentation
   (`yolov8_instance_segmentation.pt`), the largest non-face mask, then the corner extraction of
   `common/utils/alignment.py::align_image` (dilate, convex hull, `approxPolyDP`, the 5-corner fix,
   `order_points`). `card_swap.detect.quad_from_mask` is a copy of that corner logic that returns the
   corners instead of warping; `tests/test_prod_equivalence.py` checks it gives the same corners as
   production. Unlike production, the orientation model is not run: photos are used as they are (EXIF
   orientation applied).
2. **Warp the template** onto the four corners (its long side onto the card's long side). The card mask has
   rounded corners. The new card must cover at least 97% of the detected card, otherwise the sample is
   dropped, so pieces of the old document never stay visible.
3. **Transfer the light** from the real card (`--light match`, default): exposure from its bright parts,
   shading across the card, colour cast, black level, the real highlights, matched sharpness and noise,
   plus random glare or a soft shadow in some samples, then JPEG. `--light simple` is the earlier, milder
   version (shading, cast and highlights only).
4. **Map the labels**: every template polygon goes through the same homography. Samples where a label falls
   outside the photo are dropped.

## Usage

Two steps: find the card in each real photo (`cards.jsonl`), then put template cards onto it.

```bash
pip install -r requirements.txt            # numpy, opencv; + ultralytics for `detect`

# 1a. find cards from labelled TD1 MRZ lines (laptop, no models)
python -m card_swap from-mrz --photos DATASET_ROOT --labels DATASET_ROOT/v7/annotations/train_A_real.json \
    --skip-synthetic --limit 100 --out cards.jsonl

# 1b. or with the production YOLO model (GPU box / SageMaker)
export APP_VERACITY_SEGMENTATION_MODEL_PATH=/path/to/yolov8_instance_segmentation.pt
python -m card_swap detect --photos photos/ --out cards.jsonl

# 2. swap (CPU, plain image processing)
python -m card_swap swap --photos DATASET_ROOT --cards cards.jsonl --templates templates/ --out out/ \
    --per-photo 2 --vis
```

`from-mrz` options: `--categories mrz_line1,mrz_line2,mrz_line3` (COCO category names, top to bottom),
`--skip-synthetic`, `--limit N` (random N of the photos that have all 3 lines), `--seed`. `--photos` is the folder
the COCO `file_name` paths are relative to. Photos whose size differs from the COCO entry (EXIF rotation) are
skipped. Besides the quad, it stores the labelled MRZ lines as `cover_polys`; `swap` drops a sample where the new
card leaves more than 0.5% of them visible.

`swap` options: `--only CH_,FRA_` (template name filter), `--per-photo N`, `--limit N` (random N photos),
`--light match|simple`, `--min-cover 0.97`, `--rotate180-p 0.0` (share of upside-down cards), `--seed`, `--vis`.

### Inputs

`cards.jsonl`, one line per photo (written by `detect`, or by hand / from other labels):

```json
{"photo": "sub/img.jpg", "quad": [[x, y], [x, y], [x, y], [x, y]], "cls": 3, "mask_poly": [[x, y], ...]}
{"photo": "sub/other.jpg", "error": "card_not_found"}
```

`from-mrz` writes `cover_polys` (the old MRZ lines) instead of `mask_poly`.

`quad` is TL, TR, BR, BL in photo pixels. `mask_poly` (optional) is the outline of the real card, used for
the coverage check.

`templates/templates.json`, one entry per template image; the card must fill the whole image:

```json
{
  "CH_02": {"image": "CH_02.png",
            "fields": {"mrz_line1": [[37, 368], [276, 368], [276, 403], [37, 403]]},
            "corner_radius": 0.04,
            "mrz": ["..."]}
}
```

`image` defaults to `<name>.png`. Any extra keys (`mrz`, `country`, ...) are copied into the output as `meta`.
The older Swiss format `{"lines": [q1, q2, q3]}` is read as `mrz_line1..3`.

### Outputs

- `out/images/*.jpg`.
- `out/annotations.jsonl`: `image`, source `photo`, `template`, `size`, card `quad`, mapped `fields`,
  template `meta`.
- `out/vis/*.jpg` with `--vis`: the card outline (dot = template's top-left corner) and the fields.
- A count of `ok`, `rejected`, `no_card` and `card_not_id1_shape` is printed.

## Python API

```python
from card_swap import load_templates, quad_from_mask, swap_card, quad_from_mrz
tpls = load_templates("templates/")
r = swap_card(photo_bgr, quad, tpls["CH_02"], np.random.default_rng(0), orig_mask=card_mask)
r["image"], r["fields"], r["quad"]
```

Without a detector, `quad_from_mrz(photo, mrz_line_polygons, template)` finds the card from labelled MRZ lines.
With 3 lines (TD1) the card is placed by the standard TD1 geometry (`TD1_MRZ_MM`: where the MRZ sits on an
ID-1 card, measured on real photos, line pitch as in ICAO 9303), then `snap_quad_to_edges` moves each side onto
the card's straight border within a few mm of that prediction. Samples whose edges are not found, drift too far,
or give a non-ID-1 shape (a card whose MRZ is placed differently) are dropped. Other line counts use the
template's MRZ and GrabCut (how the Swiss set was made).

## Limits

- ID-1 cards only (aspect 1.3–1.9). Passports and folded documents are not handled.
- The card stays flat: no bending, and the perspective is the one in the source photo.
- Holograms and the chip are whatever the template has. The light transfer copies the real card's highlights,
  not the template's own hologram.
- Check `vis/` before training: a few samples look flatter or lighter than the scene; drop those.

## Tests

```bash
pytest                                                    # synthetic arrays only, no models
DOCVER_ML_APP=/path/to/docver-ml-app pytest tests/test_prod_equivalence.py   # corners == production
```
