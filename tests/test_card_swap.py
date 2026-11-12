"""Synthetic-array tests: no models, no real photos."""
import json

import cv2
import numpy as np
import pytest

from card_swap import Template, grabcut_quad, load_templates, quad_from_mask, regular, swap_card
from card_swap.__main__ import main
from card_swap.detect import _order_tl_tr_br_bl
from card_swap.swap import fit_orientation

W, H = 1200, 900


def card_corners(cx=600, cy=450, w=500, h=316, ang=12.0):
    """TL, TR, BR, BL of a rotated ID-1-shaped card."""
    r = np.deg2rad(ang)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (base @ R.T + [cx, cy]).astype(np.float32)


def scene(q, seed=0):
    """Dark textured table, light card with dark 'text' and a light gradient over it."""
    rng = np.random.default_rng(seed)
    img = (rng.normal(60, 12, (H, W, 3))).clip(0, 255).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 1)
    img[m > 0] = (215, 205, 200)
    for i in range(6):
        a, b = q[0] + (q[3] - q[0]) * (0.2 + 0.1 * i), q[1] + (q[2] - q[1]) * (0.2 + 0.1 * i)
        cv2.line(img, tuple((a + (b - a) * 0.1).astype(int)), tuple((a + (b - a) * 0.7).astype(int)), (40, 40, 40), 4)
    grad = np.linspace(0.75, 1.1, W, dtype=np.float32)[None, :, None]
    return (img * grad).clip(0, 255).astype(np.uint8), m


def template():
    img = np.full((540, 856, 3), (180, 160, 230), np.uint8)
    cv2.putText(img, "SPECIMEN", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (60, 30, 30), 8)
    fields = {"mrz_line1": np.float32([[40, 430], [816, 430], [816, 470], [40, 470]]),
              "mrz_line2": np.float32([[40, 480], [816, 480], [816, 520], [40, 520]])}
    return Template("t", img, fields)


def test_order_points():
    q = card_corners()
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        np.testing.assert_allclose(_order_tl_tr_br_bl(q[perm]), q, atol=1e-4)


@pytest.mark.parametrize("ang", [0.0, 8.0, -15.0, 30.0])
def test_quad_from_mask_rotated(ang):
    q = card_corners(ang=ang)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 1)
    got = quad_from_mask(m)
    assert got is not None
    # prod dilates the mask (5x5, 2 iterations): corners move outwards by a few pixels
    assert np.linalg.norm(got - _order_tl_tr_br_bl(q), axis=1).max() < 9


def test_quad_from_mask_perspective():
    q = np.float32([[400, 250], [850, 300], [820, 640], [380, 600]])
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 1)
    assert np.linalg.norm(quad_from_mask(m) - q, axis=1).max() < 9


def test_quad_from_mask_cut_corner():
    """One corner cut off: like prod, the other three corners stay exact; the cut one is approximate
    (prod restores it only when the 4-point approximation fails and a 5-gon remains)."""
    q = card_corners(ang=10)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 1)
    tr = q[1].astype(int)
    cv2.fillConvexPoly(m, np.int32([tr + [-60, -40], tr + [40, -40], tr + [40, 60]]), 0)  # chop TR corner
    got = quad_from_mask(m)
    assert got is not None
    d = np.linalg.norm(got - q, axis=1)
    assert d[[0, 2, 3]].max() < 9 and d[1] < 40


@pytest.mark.parametrize("pts,exp", [
    ([[160, 100], [500, 100], [500, 400], [100, 400], [100, 160]], [100, 100]),   # TL cut -> restored
    ([[100, 100], [500, 100], [500, 400], [160, 400], [100, 340]], [100, 400]),   # BL cut -> restored
])
def test_five_corner_restoration(pts, exp):
    from card_swap.detect import _five_to_four
    r = _order_tl_tr_br_bl(_five_to_four(np.array(pts, np.int32).reshape(-1, 1, 2)))
    assert np.abs(r - np.float32([[100, 100], [500, 100], [500, 400], [100, 400]])).max() < 1


def test_quad_from_mask_empty():
    assert quad_from_mask(np.zeros((50, 50), np.uint8)) is None


def test_regular():
    assert regular(card_corners())
    assert regular(card_corners()[[1, 2, 3, 0]])                      # portrait order is fine too
    assert not regular(card_corners(w=400, h=400))                    # square
    assert not regular(np.float32([[0, 0], [500, 0], [700, 316], [200, 316]]))  # parallelogram, 60 deg


def test_fit_orientation_long_side():
    q = card_corners()
    portrait = q[[1, 2, 3, 0]]  # starts on the short side
    np.testing.assert_allclose(fit_orientation(portrait, (540, 856)), q[[2, 3, 0, 1]])
    np.testing.assert_allclose(fit_orientation(q, (540, 856)), q)
    np.testing.assert_allclose(fit_orientation(q, (540, 856), rotate180=True), q[[2, 3, 0, 1]])


@pytest.mark.parametrize("light", ["match", "simple"])
def test_swap_card_maps_fields_and_keeps_background(light):
    q = card_corners()
    img, m = scene(q)
    tpl = template()
    r = swap_card(img, q, tpl, np.random.default_rng(0), light=light, orig_mask=m)
    assert r is not None
    out = r["image"]
    assert out.shape == img.shape and out.dtype == np.uint8
    # fields: template corners -> card corners, so a field corner maps by the same homography
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [856, 0], [856, 540], [0, 540]]), q)
    exp = cv2.perspectiveTransform(tpl.fields["mrz_line2"][None], Hm)[0]
    np.testing.assert_allclose(r["fields"]["mrz_line2"], exp, atol=1e-3)
    # background (far from the card) is untouched up to JPEG re-encoding (which smooths the pixel noise)
    far = cv2.dilate(m, np.ones((41, 41), np.uint8)) == 0
    blur = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 3)
    assert np.abs(blur(out)[far] - blur(img)[far]).mean() < 2
    # the card itself changed: the template is pinkish, the original was grey
    inside = cv2.erode(m, np.ones((21, 21), np.uint8)) > 0
    assert out[inside][:, 2].mean() - out[inside][:, 1].mean() > 20


def test_swap_card_light_follows_photo():
    """A darker photo gives a darker swapped card."""
    q = card_corners()
    img, m = scene(q)
    dark = (img * 0.5).astype(np.uint8)
    inside = cv2.erode(m, np.ones((21, 21), np.uint8)) > 0
    a = swap_card(img, q, template(), np.random.default_rng(1), orig_mask=m)["image"][inside].mean()
    b = swap_card(dark, q, template(), np.random.default_rng(1), orig_mask=m)["image"][inside].mean()
    assert b < a * 0.8


def test_swap_card_rejects_uncovered_original():
    q = card_corners()
    img, _ = scene(q)
    bigger = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(bigger, card_corners(w=600, h=380).astype(np.int32), 1)  # old card larger than the quad
    assert swap_card(img, q, template(), np.random.default_rng(0), orig_mask=bigger) is None


def test_swap_card_rejects_fields_outside():
    q = card_corners(cx=150, w=500)  # card partly out of frame on the left
    img, _ = scene(q)
    assert swap_card(img, q, template(), np.random.default_rng(0)) is None


def test_grabcut_quad_refines_rough_guess():
    q = card_corners(ang=7)
    img, _ = scene(q)
    rough = q + np.float32([[25, -20], [-15, 18], [20, 22], [-25, -15]])
    got = grabcut_quad(img, rough)
    assert got is not None
    assert np.linalg.norm(got - q, axis=1).max() < 12


def test_load_templates_and_cli(tmp_path):
    tpl = template()
    td = tmp_path / "tpl"; td.mkdir()
    cv2.imwrite(str(td / "a.png"), tpl.image)
    cv2.imwrite(str(td / "b_img.png"), tpl.image)
    json.dump({"a": {"lines": [tpl.fields["mrz_line1"].tolist(), tpl.fields["mrz_line2"].tolist()], "mrz": ["X", "Y"]},
               "b": {"image": "b_img.png", "fields": {"name": [[100, 100], [300, 100], [300, 140], [100, 140]]},
                     "corner_radius": 0.03}}, open(td / "templates.json", "w"))
    t = load_templates(td)
    assert set(t["a"].fields) == {"mrz_line1", "mrz_line2"} and t["a"].meta == {"mrz": ["X", "Y"]}
    assert t["b"].corner_radius == 0.03 and set(t["b"].fields) == {"name"}

    q = card_corners()
    img, m = scene(q)
    pd_ = tmp_path / "photos" / "sub"; pd_.mkdir(parents=True)
    cv2.imwrite(str(pd_ / "p1.jpg"), img)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    with open(tmp_path / "cards.jsonl", "w") as f:
        f.write(json.dumps({"photo": "sub/p1.jpg", "quad": q.tolist(), "mask_poly": cs[0].reshape(-1, 2).tolist()}) + "\n")
        f.write(json.dumps({"photo": "sub/p2.jpg", "error": "card_not_found"}) + "\n")
    out = tmp_path / "out"
    main(["swap", "--photos", str(tmp_path / "photos"), "--cards", str(tmp_path / "cards.jsonl"),
          "--templates", str(td), "--out", str(out), "--per-photo", "2", "--vis"])
    rows = [json.loads(l) for l in open(out / "annotations.jsonl")]
    assert len(rows) == 2
    for r in rows:
        assert (out / r["image"]).exists() and (out / "vis" / r["image"].split("/")[-1]).exists()
        assert r["photo"] == "sub/p1.jpg" and r["template"] in ("a", "b") and len(r["quad"]) == 4
