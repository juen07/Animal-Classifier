"""Inferencia multiobjeto, integracion con la base de datos y reporte.

Recibe la ruta de una imagen, detecta simultaneamente varias especies mediante
cuadros delimitadores, calcula el Top-3 de especies mas probables por cada
deteccion, consulta la base de datos veterinaria (aplicando la regla estricta)
y exporta la imagen anotada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config, database

# Colores (RGB) por nivel de riesgo para el dibujo de cuadros.
RISK_COLORS: dict[str, tuple[int, int, int]] = {
    "ALTO": (220, 20, 20),
    "MEDIO": (240, 140, 0),
    "BAJO": (230, 200, 0),
    "NINGUNO": (30, 160, 60),
    "DESCONOCIDO": (120, 120, 120),
}


@dataclass
class Detection:
    """Una deteccion individual con su analisis de riesgo."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixeles)
    top1_key: str
    top1_conf: float
    topk: list[tuple[str, float]] = field(default_factory=list)
    vet: dict = field(default_factory=dict)


def _iou(a, b) -> float:
    """Interseccion sobre union de dos cajas xyxy."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_model(model_path: Path | str | None):
    """Carga el modelo entrenado; si no existe, cae al modelo base."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = config.TRAINED_MODEL
    model_path = Path(model_path)
    if not model_path.exists():
        print(
            f"[detect] Aviso: no se encontro '{model_path}'. Usando modelo base "
            f"'{config.BASE_MODEL}' (las clases no seran las de Yucatan)."
        )
        return YOLO(config.BASE_MODEL), False
    return YOLO(str(model_path)), True


def _class_name(model, cls_id: int, custom: bool) -> str:
    """Nombre de clase segun el modelo (custom -> species keys)."""
    if custom:
        return config.ID_TO_NAME.get(cls_id, str(cls_id))
    # Modelo base: usar sus propios nombres.
    return model.names.get(cls_id, str(cls_id))


def _compute_topk(final_box, candidates, model, custom, k) -> list[tuple[str, float]]:
    """Top-k especies para una deteccion, agregando candidatos por IoU."""
    scores: dict[str, float] = {}
    for cand_box, cand_cls, cand_conf in candidates:
        if _iou(final_box, cand_box) >= 0.5:
            name = _class_name(model, int(cand_cls), custom)
            scores[name] = max(scores.get(name, 0.0), float(cand_conf))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]


def analyze_image(
    image_path: Path | str,
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    model_path: Path | str | None = None,
) -> list[Detection]:
    """Analiza una imagen y devuelve la lista de detecciones enriquecidas."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    model, custom = _load_model(model_path)

    # Pasada final (detecciones de alta confianza). Se usa NMS agnostico a la
    # clase para que un mismo objeto no genere cajas duplicadas de distintas
    # especies; las alternativas se reportan luego en el Top-k.
    final = model.predict(
        source=str(image_path),
        conf=cfg.conf,
        iou=cfg.iou,
        agnostic_nms=True,
        verbose=False,
    )[0]

    # Pasada de candidatos (baja confianza) para extraer alternativas Top-k.
    cand_res = model.predict(
        source=str(image_path), conf=0.01, iou=cfg.iou, verbose=False
    )[0]
    candidates = []
    if cand_res.boxes is not None:
        for box in cand_res.boxes:
            candidates.append(
                (
                    tuple(float(v) for v in box.xyxy[0].tolist()),
                    int(box.cls[0]),
                    float(box.conf[0]),
                )
            )

    detections: list[Detection] = []
    if final.boxes is None or len(final.boxes) == 0:
        return detections

    for box in final.boxes:
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        top1 = _class_name(model, cls_id, custom)

        topk = _compute_topk(xyxy, candidates, model, custom, cfg.topk)
        # Garantizar que la clase ganadora encabece el Top-k.
        if not topk or topk[0][0] != top1:
            topk = [(top1, conf)] + [t for t in topk if t[0] != top1]
            topk = topk[: cfg.topk]

        vet = database.describe_toxicity(top1) if custom else {
            "species_key": top1,
            "common_name": top1,
            "risk_level": "DESCONOCIDO",
            "message": database.NO_RECORD_MESSAGE,
            "has_info": False,
        }

        detections.append(
            Detection(bbox=xyxy, top1_key=top1, top1_conf=conf, topk=topk, vet=vet)
        )

    return detections


def print_report(image_path: Path | str, detections: list[Detection]) -> None:
    """Imprime en consola el reporte textual con el Top-3 por deteccion."""
    print("=" * 70)
    print(f"REPORTE DE ANALISIS -> {image_path}")
    print("=" * 70)

    if not detections:
        print("No se detectaron animales en la imagen.")
        print("=" * 70)
        return

    print(f"Animales detectados: {len(detections)}\n")
    for i, det in enumerate(detections, 1):
        common = det.vet.get("common_name", det.top1_key)
        risk = det.vet.get("risk_level", "DESCONOCIDO")
        print(f"[Deteccion #{i}] {common}  (riesgo: {risk})")
        x1, y1, x2, y2 = (round(v, 1) for v in det.bbox)
        print(f"  Ubicacion (bbox): ({x1}, {y1}) - ({x2}, {y2})")

        print("  Top 3 especies mas probables:")
        for rank, (name, score) in enumerate(det.topk, 1):
            sp = config.KEY_TO_SPECIES.get(name)
            label = sp.common_name if sp else name
            print(f"    {rank}. {label:<28} {score * 100:5.1f}%")

        # Regla estricta: mensaje obligatorio o informacion completa.
        if det.vet.get("has_info"):
            print(f"  Toxicidad/danos: {det.vet.get('toxicity', '')}")
            print(f"  Sintomas: {det.vet.get('symptoms', '')}")
            print(f"  Primeros auxilios: {det.vet.get('first_aid', '')}")
        else:
            print(f"  {det.vet.get('message', database.NO_RECORD_MESSAGE)}")
        print()
    print("=" * 70)


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(
    image_path: Path | str,
    detections: list[Detection],
    out_path: Path | str | None = None,
) -> Path:
    """Dibuja los cuadros delimitadores y niveles de riesgo sobre la imagen."""
    image_path = Path(image_path)
    if out_path is None:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.OUTPUTS_DIR / f"{image_path.stem}_analizado{image_path.suffix}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(max(14, img.width // 45))

    for det in detections:
        risk = det.vet.get("risk_level", "DESCONOCIDO") or "DESCONOCIDO"
        color = RISK_COLORS.get(risk.upper(), RISK_COLORS["DESCONOCIDO"])
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        common = det.vet.get("common_name", det.top1_key)
        label = f"{common} [{risk}] {det.top1_conf * 100:.0f}%"

        # Fondo del texto.
        try:
            tb = draw.textbbox((x1, y1), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:  # noqa: BLE001
            tw, th = len(label) * 7, 14
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)

    img.save(out_path)
    print(f"[detect] Imagen anotada exportada en {out_path}")
    return out_path


def run(
    image_path: Path | str,
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    model_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> list[Detection]:
    """Flujo completo de inferencia: analizar, reportar y exportar imagen."""
    detections = analyze_image(image_path, cfg=cfg, model_path=model_path)
    print_report(image_path, detections)
    draw_detections(image_path, detections, out_path=out_path)
    return detections
