"""
predict.py
-----------
Eğitilmiş diş röntgeni nesne tespiti modeli için bağımsız çıkarım betiği.

Örnek kullanım:
    python predict.py --image test_pics/0001.jpg
    python predict.py --image test_pics/0001.jpg --conf 0.30 --output docs/results/example.jpg
    python predict.py --image test_pics/0001.jpg --no-show

Not: Bu proje klinik karar verme amacıyla değil, portföy ve araştırma prototipi
olarak hazırlanmıştır. Model çıktıları hekim değerlendirmesinin yerine geçmez.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dental_ai.inference import (  # noqa: E402
    get_default_model_path,
    predict_image,
    save_annotated_image,
    summarize_detections,
    validate_confidence_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını okur."""
    parser = argparse.ArgumentParser(
        description="Eğitilmiş diş röntgeni tespit modelini tek bir görsel üzerinde çalıştırır."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Girdi röntgen görselinin yolu (.jpg/.jpeg/.png/.webp).",
    )
    parser.add_argument(
        "--weights",
        "-w",
        default=str(get_default_model_path()),
        help="Model ağırlık dosyasının yolu. Varsayılan: models/best.pt",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.50,
        help="Minimum güven eşiği. 0 ile 1 arasında olmalıdır. Varsayılan: 0.50",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="İşaretlenmiş çıktı görselinin kaydedileceği yol. Varsayılan: <görsel>_result.<uzantı>",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Sonuç görselini işletim sisteminin varsayılan görüntüleyicisinde açma.",
    )
    return parser.parse_args()


def print_report(model, result, image_path: str | Path, confidence_threshold: float) -> None:
    """Tahminleri terminalde okunabilir bir rapor olarak gösterir."""
    detections = summarize_detections(model, result)

    print("\n" + "=" * 68)
    print("DENTAL AI RÖNTGEN ANALİZ RAPORU")
    print(f"Dosya: {Path(image_path).name}")
    print(f"Güven eşiği: {confidence_threshold:.2f}")
    print("-" * 68)

    if not detections:
        print("Sonuç: Belirtilen güven eşiğinin üzerinde bulgu tespit edilmedi.")
    else:
        for index, detection in enumerate(detections, start=1):
            print(
                f"{index}. {detection.class_name_tr} ({detection.class_name}) | "
                f"güven: {detection.confidence:.2f} | öneri: {detection.recommendation}"
            )

    print("=" * 68)
    print("UYARI: Bu çıktı klinik tanı değildir; hekim değerlendirmesinin yerine geçmez.\n")


def open_image_with_default_viewer(image_path: str | Path) -> None:
    """Kaydedilen sonucu varsayılan görsel görüntüleyiciyle açar."""
    path = str(image_path)
    system = platform.system()

    try:
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", path], check=True)
        else:
            subprocess.run(["xdg-open", path], check=True)
        logger.info("Sonuç görseli açıldı: %s", path)
    except Exception as exc:  # pragma: no cover - GUI ortamına bağlıdır.
        logger.warning("Sonuç görseli otomatik açılamadı (%s). Dosya yolu: %s", exc, path)


def main() -> None:
    """CLI akışını çalıştırır."""
    args = parse_args()

    try:
        confidence_threshold = validate_confidence_threshold(args.conf)
        logger.info("Model yükleniyor ve görsel analiz ediliyor...")
        model, result = predict_image(
            image_path=args.image,
            model_path=args.weights,
            confidence_threshold=confidence_threshold,
        )
        print_report(model, result, args.image, confidence_threshold)
        output_path = save_annotated_image(result, args.image, args.output)
        logger.info("İşaretlenmiş görsel kaydedildi: %s", output_path)

        if not args.no_show:
            open_image_with_default_viewer(output_path)

    except Exception as exc:
        logger.error("İşlem başarısız: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
