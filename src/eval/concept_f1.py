"""F1 de Concept Detection — réplica de la métrica oficial de ImageCLEF.

Metodología oficial:
- Para cada imagen se construyen y_true / y_pred binarios sobre el vocabulario
  = union(conceptos GT, conceptos predichos) de esa imagen.
- Se calcula F1 con scikit-learn, averaging 'binary'.
- Se promedia sobre todas las imágenes del test.
- Caso borde: si GT y predicción están ambos vacíos -> F1 = 1.0 (acierto perfecto).

Score primario: todos los conceptos.
Score secundario: filtrar GT y predicción al conjunto de conceptos anotados a mano
                  antes de repetir el cálculo (pasar `manual_concepts`).

Formato de fichero esperado (CSV, una fila por imagen):
    <image_id><sep><CUI1;CUI2;CUI3>
donde <sep> es tab o coma, y los conceptos van separados por ';'.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import f1_score


def load_concept_file(path: str | Path) -> dict[str, set[str]]:
    """Carga un fichero de conceptos -> {image_id: {CUI, ...}}."""
    result: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.reader(fh, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            image_id = row[0].strip()
            if image_id.lower() in {"id", "image_id", "imageid"}:
                continue  # cabecera
            concepts_raw = row[1] if len(row) > 1 else ""
            concepts = {c.strip() for c in concepts_raw.split(";") if c.strip()}
            result[image_id] = concepts
    return result


def _image_f1(gt: set[str], pred: set[str]) -> float:
    vocab = sorted(gt | pred)
    if not vocab:
        return 1.0  # ambos vacíos -> acierto perfecto
    y_true = [1 if c in gt else 0 for c in vocab]
    y_pred = [1 if c in pred else 0 for c in vocab]
    return f1_score(y_true, y_pred, average="binary", zero_division=0)


def compute_concept_f1(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    manual_concepts: set[str] | None = None,
) -> float:
    """F1 promedio sobre las imágenes de `ground_truth`.

    Si `manual_concepts` se pasa, filtra GT y predicción a ese conjunto (score secundario).
    Imágenes sin predicción se tratan como predicción vacía.
    """
    scores = []
    for image_id, gt in ground_truth.items():
        pred = predictions.get(image_id, set())
        if manual_concepts is not None:
            gt = gt & manual_concepts
            pred = pred & manual_concepts
        scores.append(_image_f1(gt, pred))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="F1 de Concept Detection (ImageCLEF)")
    ap.add_argument("--pred", required=True, help="CSV de predicciones")
    ap.add_argument("--gt", required=True, help="CSV de ground truth")
    ap.add_argument("--manual", help="CSV/txt opcional con conceptos anotados a mano (score secundario)")
    args = ap.parse_args()

    predictions = load_concept_file(args.pred)
    ground_truth = load_concept_file(args.gt)

    primary = compute_concept_f1(predictions, ground_truth)
    print(f"F1 (primary) : {primary:.4f}")

    if args.manual:
        manual: set[str] = set()
        for concepts in load_concept_file(args.manual).values():
            manual |= concepts
        if not manual:  # el fichero podría ser una lista plana de CUIs
            with open(args.manual, encoding="utf-8") as fh:
                manual = {ln.strip() for ln in fh if ln.strip()}
        secondary = compute_concept_f1(predictions, ground_truth, manual_concepts=manual)
        print(f"F1 (secondary): {secondary:.4f}")


if __name__ == "__main__":
    main()
