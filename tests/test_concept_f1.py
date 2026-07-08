"""Tests de la métrica F1 de Concept Detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.concept_f1 import _image_f1, compute_concept_f1  # noqa: E402


def test_perfect_match():
    assert _image_f1({"C1", "C2"}, {"C1", "C2"}) == 1.0


def test_total_mismatch():
    assert _image_f1({"C1", "C2"}, {"C3", "C4"}) == 0.0


def test_partial_overlap():
    # gt={C1,C2,C3}, pred={C1,C2,C4}: TP=2, FP=1, FN=1
    # precision=2/3, recall=2/3, F1=2/3
    assert abs(_image_f1({"C1", "C2", "C3"}, {"C1", "C2", "C4"}) - 2 / 3) < 1e-9


def test_both_empty_is_perfect():
    assert _image_f1(set(), set()) == 1.0


def test_empty_prediction_scores_zero():
    assert _image_f1({"C1"}, set()) == 0.0


def test_empty_gt_scores_zero():
    assert _image_f1(set(), {"C1"}) == 0.0


def test_average_over_images():
    gt = {"img1": {"C1", "C2"}, "img2": {"C3"}}
    pred = {"img1": {"C1", "C2"}, "img2": set()}  # img1 perfecto, img2 cero
    assert abs(compute_concept_f1(pred, gt) - 0.5) < 1e-9


def test_missing_prediction_treated_as_empty():
    gt = {"img1": {"C1"}}
    pred = {}  # sin prediccion para img1
    assert compute_concept_f1(pred, gt) == 0.0


def test_secondary_filters_to_manual():
    gt = {"img1": {"C1", "C2"}}       # C1 manual, C2 no
    pred = {"img1": {"C1"}}
    # Con filtro manual={C1}: gt={C1}, pred={C1} -> F1=1.0
    assert compute_concept_f1(pred, gt, manual_concepts={"C1"}) == 1.0


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
