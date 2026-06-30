"""
Diş röntgeni nesne tespiti modeli için ortak çıkarım yardımcıları.

Bu modül hem `predict.py` komut satırı betiği hem de Jupyter notebook tarafından
kullanılır. Böylece model yükleme, doğrulama, tahmin özeti ve görsel
oluşturma mantığı tek bir yerde tutulur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_INFO: dict[str, dict[str, str]] = {
    "Cavity": {
        "tr_name": "Çürük",
        "note": "Restoratif tedavi ve çürük derinliği açısından klinik değerlendirme önerilir.",
    },
    "Implant": {
        "tr_name": "İmplant",
        "note": "Kemik seviyesi ve peri-implant doku sağlığı rutin kontrolde değerlendirilmelidir.",
    },
    "Fillings": {
        "tr_name": "Dolgu",
        "note": "Mevcut dolgunun kenar uyumu ve sekonder çürük açısından kontrolü önerilir.",
    },
    "Impacted Tooth": {
        "tr_name": "Gömülü Diş",
        "note": "Komşu anatomik yapılarla ilişkisi için çene cerrahisi değerlendirmesi önerilir.",
    },
}


@dataclass(frozen=True)
class DetectionSummary:
    """Tek bir nesne tespitinin sadeleştirilmiş özeti."""

    class_id: int
    class_name: str
    class_name_tr: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    recommendation: str

    @property
    def confidence_percent(self) -> float:
        """Güven skorunu yüzde formatına dönüştürür."""
        return self.confidence * 100

    def as_dict(self) -> dict[str, Any]:
        """Notebook içinde tablo göstermek için sözlük çıktısı üretir."""
        x1, y1, x2, y2 = self.box_xyxy
        return {
            "Sınıf": self.class_name_tr,
            "Model Etiketi": self.class_name,
            "Güven": round(self.confidence, 4),
            "Güven (%)": round(self.confidence_percent, 2),
            "Kutu (x1, y1, x2, y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
            "Öneri": self.recommendation,
        }


def get_project_root() -> Path:
    """`src/dental_ai` konumundan proje kök dizinini bulur."""
    return Path(__file__).resolve().parents[2]


def get_default_model_path() -> Path:
    """Repoda yer alan varsayılan eğitilmiş model ağırlığının yolunu döndürür."""
    return get_project_root() / "models" / "best.pt"


def validate_image_path(image_path: str | Path) -> Path:
    """Görsel yolunu ve dosya uzantısını doğrular."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Görsel dosyası bulunamadı: {path}")
    if not path.is_file():
        raise ValueError(f"Girdi bir dosya değil: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Desteklenmeyen dosya formatı: {path.suffix}. Desteklenen formatlar: {allowed}")
    return path


def validate_model_path(model_path: str | Path) -> Path:
    """Model ağırlık dosyasının varlığını doğrular."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Model ağırlık dosyası bulunamadı: {path}. "
            "Beklenen dosya: models/best.pt"
        )
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Model ağırlığı .pt uzantılı olmalıdır: {path}")
    return path


def validate_confidence_threshold(confidence_threshold: float) -> float:
    """Güven eşiğinin geçerli aralıkta olup olmadığını kontrol eder."""
    threshold = float(confidence_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("Güven eşiği 0 ile 1 arasında olmalıdır.")
    return threshold


def load_model(model_path: str | Path | None = None):
    """Ultralytics YOLO modelini yükler."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "'ultralytics' paketi kurulu değil. Kurulum için: pip install -r requirements.txt"
        ) from exc

    path = validate_model_path(model_path or get_default_model_path())
    return YOLO(str(path))


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.50,
):
    """
    Tek bir röntgen görseli üzerinde nesne tespiti çalıştırır.

    Returns
    -------
    tuple
        `(model, result)` biçiminde Ultralytics model nesnesi ve ilk sonuç döner.
    """
    image = validate_image_path(image_path)
    threshold = validate_confidence_threshold(confidence_threshold)
    model = load_model(model_path)
    results = model(str(image), conf=threshold, verbose=False)
    return model, results[0]


def summarize_detections(model: Any, result: Any) -> list[DetectionSummary]:
    """Ultralytics sonuç nesnesinden okunabilir tespit listesi üretir."""
    summaries: list[DetectionSummary] = []

    if result.boxes is None or len(result.boxes) == 0:
        return summaries

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names.get(class_id, str(class_id))
        class_info = CLASS_INFO.get(
            class_name,
            {"tr_name": class_name, "note": "Hekim tarafından detaylı klinik muayene önerilir."},
        )
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        summaries.append(
            DetectionSummary(
                class_id=class_id,
                class_name=class_name,
                class_name_tr=class_info["tr_name"],
                confidence=confidence,
                box_xyxy=(x1, y1, x2, y2),
                recommendation=class_info["note"],
            )
        )

    return sorted(summaries, key=lambda item: item.confidence, reverse=True)


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    """Bir görseli OpenCV ile okuyup Matplotlib uyumlu RGB dizisi olarak döndürür."""
    path = validate_image_path(image_path)
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Görsel okunamadı: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render_annotated_image_rgb(result: Any) -> np.ndarray:
    """Model sonucunu kutu/etiket çizilmiş RGB görsel olarak üretir."""
    annotated_bgr = result.plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def save_annotated_image(
    result: Any,
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Kutu ve etiket çizilmiş tahmin görselini diske kaydeder."""
    source_path = validate_image_path(image_path)
    if output_path is None:
        output_path = source_path.with_name(f"{source_path.stem}_result{source_path.suffix}")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    annotated_bgr = result.plot()
    success = cv2.imwrite(str(output), annotated_bgr)
    if not success:
        raise IOError(f"Sonuç görseli kaydedilemedi: {output}")
    return output
