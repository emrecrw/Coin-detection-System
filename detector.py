from dataclasses import dataclass
from typing import List, Optional

from src.config import DEFAULT_MODEL_PATH, value_from_label
from src.total_calculator import CoinDetection, calculate_total, CoinTotalResult


# Seuils par défaut
CONF_BASE = 0.10   # confiance minimale pour considérer un objet comme une pièce
# au-dessus, on fait confiance à la classe euro prédite
# (entre CONF_BASE et CONF_EURO -> pièce étrangère / inconnue)
CONF_EURO = 0.25                   


@dataclass
class DetectedCoin:
    """Une pièce détectée sur l'image, euro ou étrangère."""
    class_name: str          # nom de classe prédit par le modèle
    confidence: float
    bbox: List[float]        # [x_min, y_min, x_max, y_max]
    is_foreign: bool         # True si pièce étrangère / non reconnue
    value: float             # valeur en euros (0.0 si étrangère ou parasite)


@dataclass
class PredictionResult:
    """Résultat complet de l'analyse d'une image."""
    coins: List[DetectedCoin]            # toutes les pièces (euros + étrangères)
    total_result: CoinTotalResult        # détail et somme des euros uniquement
    n_foreign: int                       # nombre de pièces étrangères

    @property
    def total(self) -> float:
        return self.total_result.total


class CoinPredictor:
    """Charge le modèle une fois, puis l'applique à plusieurs images."""

    def __init__(self, model_path=None, conf_base: float = CONF_BASE,
                 conf_euro: float = CONF_EURO):
        from ultralytics import YOLO

        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.conf_base = conf_base
        self.conf_euro = conf_euro
        self.model = YOLO(self.model_path)

    def _name_for(self, cid: int) -> Optional[str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return names.get(cid)
        if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            return names[cid]
        return None

    def predict(self, image, imgsz: int = 640) -> PredictionResult:
        """
        Analyse une image (chemin ou tableau BGR) et renvoie le résultat.
        """
        result = self.model.predict(
            image, conf=self.conf_base, imgsz=imgsz, verbose=False
        )[0]

        coins: List[DetectedCoin] = []
        euro_detections: List[CoinDetection] = []

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()

            for cid, conf, box in zip(cls_ids, confs, xyxy):
                conf = float(conf)
                bbox = box.tolist()
                name = self._name_for(int(cid)) or "?"
                euro_value = value_from_label(name)

                # C'est la confiance suffisante 
                if conf >= self.conf_euro and euro_value > 0:
                    coins.append(DetectedCoin(
                        class_name=name, confidence=conf, bbox=bbox,
                        is_foreign=False, value=euro_value,
                    ))
                    euro_detections.append(CoinDetection(
                        class_name=name, confidence=conf, bbox=bbox,
                    ))
                else:
                    # Objet de forme de pièce non reconnu comme euro
                    coins.append(DetectedCoin(
                        class_name=name, confidence=conf, bbox=bbox,
                        is_foreign=True, value=0.0,
                    ))

        total_result = calculate_total(euro_detections)
        n_foreign = sum(1 for c in coins if c.is_foreign)

        return PredictionResult(
            coins=coins,
            total_result=total_result,
            n_foreign=n_foreign,
        )


# Libellés et format de la somme
def format_value_label(value: float) -> str:
    """Libellé lisible d'une valeur de pièce (ex. '20 centimes', '1 euro')."""
    if value >= 1.0:
        n = int(round(value))
        return f"{n} euro" if n == 1 else f"{n} euros"
    cents = int(round(value * 100))
    return f"{cents} centime" if cents == 1 else f"{cents} centimes"


def format_total(total: float) -> str:
    """Met en forme un total : '4 euros 20 centimes', '95 centimes', etc."""
    cents_total = int(round(total * 100))
    euros = cents_total // 100
    cents = cents_total % 100

    parts = []
    if euros > 0:
        parts.append(f"{euros} euro" if euros == 1 else f"{euros} euros")
    if cents > 0:
        parts.append(f"{cents} centime" if cents == 1 else f"{cents} centimes")
    if not parts:
        return "0 euro"
    return " ".join(parts)


def format_summary(result: PredictionResult) -> str:
    """
    Construit le récapitulatif texte de l'analyse.
    """
    counts = result.total_result.counts_by_class
    if not counts and result.n_foreign == 0:
        return "Aucune pièce détectée."

    # Tri par valeur croissante, comme une caisse
    by_value = sorted(counts.items(), key=lambda kv: value_from_label(kv[0]))

    rows = []
    label_width = 0
    for cls, n in by_value:
        label = format_value_label(value_from_label(cls))
        rows.append((label, n))
        label_width = max(label_width, len(label))

    lines = [f"{label.ljust(label_width)} : {n}" for label, n in rows]

    if result.n_foreign > 0:
        if result.n_foreign == 1:
            lines.append("1 pièce étrangère (non comptée)")
        else:
            lines.append(f"{result.n_foreign} pièces étrangères (non comptées)")

    sep = "-" * max(16, label_width + 6)
    lines.append(sep)
    lines.append(f"SOMME : {format_total(result.total)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python -m src.predictor <image>")
        sys.exit(1)

    predictor = CoinPredictor()
    res = predictor.predict(sys.argv[1])
    print(format_summary(res))
