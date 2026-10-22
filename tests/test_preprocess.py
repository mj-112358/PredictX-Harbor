import numpy as np
import pytest

from sweepscreen.preprocess import annotation_mask, crop_to_content, letterbox, normalize_intensity, prepare, remove_annotations
from sweepscreen.preprocess.augment import DEGRADATIONS, LEVELS, make, random_augment
from tests.conftest import synthetic_frame


def test_crop_finds_field_of_view(rng):
    img, _ = synthetic_frame(rng, h=200, w=260)
    c = crop_to_content(img)
    assert abs(c.x0 - 20) <= 3 and abs(c.y0 - 20) <= 3 and abs(c.x1 - 240) <= 3 and abs(c.y1 - 180) <= 3


def test_crop_excludes_ui_text(rng):
    img, _ = synthetic_frame(rng, ui=True)
    c = crop_to_content(remove_annotations(img))
    assert c.y0 >= 15            # text sits in rows 0-14


def test_annotation_mask_marks_colour_and_text(rng):
    img, _ = synthetic_frame(rng, ui=True, colour=True)
    m = annotation_mask(img)
    assert m[60, 75] == 1          # yellow caliper line
    assert m[:15, :60].sum() > 0   # white text
    assert m[100, 130] == 0        # tissue


def test_crop_boxes_shift_and_clip():
    from sweepscreen.preprocess.crop import Crop

    c = Crop(10, 20, 110, 120)
    assert c.boxes([(15, 25, 50, 60), (0, 0, 5, 5), (100, 100, 200, 200)]) == [(5, 5, 40, 40), (90, 80, 100, 100)]


def test_letterbox_box_roundtrip():
    img = np.zeros((100, 200), np.float32)
    out, lb = letterbox(img, 224)
    assert out.shape == (224, 224)
    b = lb.boxes([(50, 20, 150, 80)])[0]
    back = lb.invert_box(b)
    assert all(abs(x - y) <= 1 for x, y in zip(back, (50, 20, 150, 80)))


def test_normalize_range():
    g = np.arange(256, dtype=np.uint8).reshape(16, 16)
    n = normalize_intensity(g)
    assert n.min() == 0 and n.max() == 1 and n.dtype == np.float32


def test_prepare_keeps_lesion_box(rng):
    img, mask = synthetic_frame(rng, center=(130, 100))
    ys, xs = np.nonzero(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    p = prepare(img, [box], size=224)
    assert p.image.shape == (224, 224) and len(p.boxes) == 1
    x0, y0, x1, y1 = p.boxes[0]
    assert p.image[y0:y1, x0:x1].mean() < p.image.mean()      # lesion is darker than average


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_each_degradation_runs_at_every_level(rng, name):
    img, mask = synthetic_frame(rng)
    g = img[20:180, 20:240].copy()
    for level in LEVELS[name]:
        out, boxes = make(name, level)(g, [(50, 50, 100, 100)])
        assert out.dtype == np.uint8 and out.shape[1] == g.shape[1]
        if name != "depth_crop":
            assert out.shape == g.shape and boxes == [(50, 50, 100, 100)]


def test_depth_crop_clips_boxes():
    g = np.full((100, 50), 100, np.uint8)
    out, boxes = DEGRADATIONS["depth_crop"](g, [(0, 60, 10, 90), (0, 10, 10, 20)], 0.25, None)
    assert out.shape == (75, 50) and boxes == [(0, 60, 10, 75), (0, 10, 10, 20)]


def test_speckle_mean_preserving(rng):
    g = np.full((200, 200), 100, np.uint8)
    out, _ = DEGRADATIONS["speckle"](g, [], 0.3, rng)
    assert abs(out.mean() - 100) < 2 and out.std() > 10


def test_random_augment_flip_mirrors_boxes():
    g = np.zeros((10, 20), np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(20):
        out, boxes = random_augment(g, [(2, 1, 5, 4)], rng, p_each=0.0)
        assert boxes in ([(2, 1, 5, 4)], [(15, 1, 18, 4)])
