"""
Lecture des annotations au format LabelMe (.json) comme vérité terrain.

Le format LabelMe décrit chaque pièce par un rectangle (deux coins) et un label
en français (ex. "50_centimes", "2_euros"). Ce module convertit ces annotations
en comptes par valeur et en total euros, pour comparaison avec les prédictions
du modèle.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.config import value_from_label


@dataclass
class GroundTruthBox:
    """Une pièce annotée."""
    label: str           # label brut du JSON (ex. "50_centimes")
    value: float         # valeur en euros déduite du label
    bbox: List[float]    # [x_min, y_min, x_max, y_max]


@dataclass
class GroundTruth:
    """Vérité terrain pour une image."""
    image_path: str
    image_width: int
    image_height: int
    boxes: List[GroundTruthBox] = field(default_factory=list)

    @property
    def counts_by_value(self) -> Dict[float, int]:
        """Nombre de pièces par valeur (ignore les labels non reconnus -> 0.0)."""
        return dict(Counter(b.value for b in self.boxes if b.value > 0))

    @property
    def total(self) -> float:
        """Total annoté en euros."""
        return round(sum(b.value for b in self.boxes), 2)

    @property
    def n_coins(self) -> int:
        """Nombre de pièces reconnues (valeur > 0)."""
        return sum(1 for b in self.boxes if b.value > 0)

    @property
    def unknown_labels(self) -> List[str]:
        """Labels présents dans le JSON mais non reconnus comme une pièce."""
        return sorted({b.label for b in self.boxes if b.value == 0.0})


def _normalize_points(points: List[List[float]]) -> List[float]:
    """
    LabelMe stocke un rectangle par deux coins quelconques.
    On renvoie [x_min, y_min, x_max, y_max] trié.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_ground_truth(json_path) -> GroundTruth:
    """Charge un fichier d'annotation LabelMe et retourne un GroundTruth."""
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes: List[GroundTruthBox] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        points = shape.get("points", [])
        if not points:
            continue
        bbox = _normalize_points(points)
        boxes.append(
            GroundTruthBox(
                label=label,
                value=value_from_label(label),
                bbox=bbox,
            )
        )

    return GroundTruth(
        image_path=data.get("imagePath", json_path.stem),
        image_width=int(data.get("imageWidth", 0) or 0),
        image_height=int(data.get("imageHeight", 0) or 0),
        boxes=boxes,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python -m src.labelme_parser <fichier.json>")
        sys.exit(1)

    gt = load_ground_truth(sys.argv[1])
    print(f"Image       : {gt.image_path} ({gt.image_width}x{gt.image_height})")
    print(f"Pièces      : {gt.n_coins}")
    print("Détail par valeur :")
    for value in sorted(gt.counts_by_value, reverse=True):
        n = gt.counts_by_value[value]
        print(f"   {value:.2f}EUR x {n} = {value * n:.2f}EUR")
    if gt.unknown_labels:
        print(f"Labels non reconnus : {gt.unknown_labels}")
    print(f"TOTAL annoté : {gt.total:.2f}EUR")
