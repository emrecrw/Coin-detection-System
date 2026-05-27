from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from src.config import COIN_VALUES, CLASS_NAMES, IGNORED_CLASSES, value_from_label


@dataclass
class CoinDetection:
    class_name: str
    confidence: float
    bbox: Sequence[float]

    @property
    def value(self) -> float:
        # Déduit la valeur directement du label (gère FR/EN, peu importe data.yaml)
        return value_from_label(self.class_name)

    @property
    def is_ignored(self) -> bool:
        return self.value == 0.0


@dataclass
class CoinTotalResult:
    counts_by_class: dict
    value_by_class: dict
    total: float
    detections: List[CoinDetection]
    ignored_detections: List[CoinDetection]

    def pretty_print(self) -> str:
        if not self.counts_by_class and not self.ignored_detections:
            return "Aucune pièce détectée."

        lines = ["Pièces détectées :"]
        sorted_classes = sorted(
            self.counts_by_class.keys(),
            key=lambda c: value_from_label(c),
            reverse=True,
        )
        for cls in sorted_classes:
            count = self.counts_by_class[cls]
            subtotal = self.value_by_class[cls]
            lines.append(
                f"  - {_format_class_label(cls)} : {count} ({subtotal:.2f}€)"
            )

        if self.ignored_detections:
            n = len(self.ignored_detections)
            lines.append("")
            lines.append(f"{n} détection(s) ignorée(s) (classe(s) parasite(s)).")

        lines.append("")
        lines.append(f"TOTAL : {self.total:.2f}€")
        return "\n".join(lines)


def _format_class_label(class_name: str) -> str:
    """Format lisible pour l'affichage."""
    n = class_name.lower()
    if "1-euro" in n or n in ("1e", "1euro"):
        return "1€"
    if "2-euro" in n or n in ("2e", "2euro"):
        return "2€"
    # Centimes : essayer d'extraire le nombre
    for digit in ("1", "2", "5", "10", "20", "50"):
        if f"{digit}-cent" in n or n == f"{digit}c":
            return f"{digit}c"
    return class_name


def calculate_total(
    detections: Iterable[CoinDetection],
    conf_threshold: float = 0.0,
) -> CoinTotalResult:
    """Calcule le total à partir d'une liste de détections."""
    filtered = [d for d in detections if d.confidence >= conf_threshold]

    valid = [d for d in filtered if not d.is_ignored]
    ignored = [d for d in filtered if d.is_ignored]

    counts = Counter(d.class_name for d in valid)
    value_by_class = {
        cls: round(count * value_from_label(cls), 2)
        for cls, count in counts.items()
    }
    total = round(sum(value_by_class.values()), 2)

    return CoinTotalResult(
        counts_by_class=dict(counts),
        value_by_class=value_by_class,
        total=total,
        detections=valid,
        ignored_detections=ignored,
    )


def from_yolo_results(yolo_result, conf_threshold: float = 0.0) -> CoinTotalResult:
    """Convertit un résultat Ultralytics en CoinTotalResult.

    On lit les noms de classes embarqués dans le modèle (`yolo_result.names`)
    plutôt que ceux de data.yaml, car ils peuvent différer du dépôt.
    """
    detections = []
    if yolo_result is None or yolo_result.boxes is None:
        return calculate_total([], conf_threshold)

    model_names = getattr(yolo_result, "names", None)

    def name_for(cid: int):
        if isinstance(model_names, dict):
            return model_names.get(cid)
        if isinstance(model_names, (list, tuple)) and 0 <= cid < len(model_names):
            return model_names[cid]
        if 0 <= cid < len(CLASS_NAMES):
            return CLASS_NAMES[cid]
        return None

    boxes = yolo_result.boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    for cid, conf, box in zip(cls_ids, confs, xyxy):
        cname = name_for(int(cid))
        if cname is None:
            continue
        detections.append(
            CoinDetection(
                class_name=cname,
                confidence=float(conf),
                bbox=box.tolist(),
            )
        )

    return calculate_total(detections, conf_threshold)