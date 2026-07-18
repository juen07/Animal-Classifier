"""Construccion del dataset en formato YOLO.

Las imagenes descargadas por el scraper no traen anotaciones espaciales. Este
modulo realiza un auto-etiquetado: usa un modelo YOLO preentrenado para ubicar
el objeto principal de cada imagen y le asigna la clase de la especie (conocida
por la carpeta de origen). Cuando el modelo no encuentra nada, se usa la imagen
completa como cuadro delimitador (etiquetado debil).

Finalmente separa el dataset en train/val y genera el archivo YAML que
Ultralytics necesita para el entrenamiento.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageChops

from . import config

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _content_bbox(image_path: Path) -> tuple[float, float, float, float] | None:
    """Estima un cuadro delimitador ajustado al sujeto (region no-fondo).

    Compara la imagen contra el color de fondo estimado de las esquinas y
    devuelve el bbox normalizado (x_center, y_center, w, h) del contenido
    saliente. Sirve como etiquetado debil mucho mas ajustado que usar la
    imagen completa. Devuelve ``None`` si no logra aislar contenido.
    """
    try:
        with Image.open(image_path) as im:
            img = im.convert("RGB")
    except OSError:
        return None

    w, h = img.size
    if w < 4 or h < 4:
        return None

    # Color de fondo estimado (promedio de las cuatro esquinas).
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    bg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))

    bg_img = Image.new("RGB", img.size, bg)
    diff = ImageChops.difference(img, bg_img).convert("L")
    mask = diff.point(lambda p: 255 if p > 25 else 0)
    box = mask.getbbox()
    if box is None:
        return None

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Descartar cajas casi vacias o casi completas (poco informativas).
    if bw <= 2 or bh <= 2:
        return None
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return (xc, yc, bw / w, bh / h)


def _iter_species_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Agrupa las rutas de imagen por species_key (nombre de carpeta)."""
    result: dict[str, list[Path]] = {}
    for species in config.SPECIES:
        folder = raw_dir / species.key
        if not folder.exists():
            continue
        imgs = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        if imgs:
            result[species.key] = imgs
    return result


def _autolabel_boxes(model, image_path: Path, class_id: int, conf: float) -> list[str]:
    """Genera lineas de etiqueta YOLO para una imagen.

    Cada linea: ``<class_id> <x_center> <y_center> <width> <height>`` (normalizado).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except OSError:
        return []

    lines: list[str] = []
    if model is not None:
        try:
            results = model.predict(
                source=str(image_path), conf=conf, verbose=False
            )
        except Exception:  # noqa: BLE001
            results = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes.xywhn:
                x, y, w, h = (float(v) for v in box.tolist())
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if not lines:
        # Fallback 1: cuadro ajustado al contenido saliente (region no-fondo).
        cbox = _content_bbox(image_path)
        if cbox is not None:
            x, y, w, h = cbox
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        else:
            # Fallback 2: etiquetado debil con la imagen completa.
            lines.append(f"{class_id} 0.5 0.5 1.0 1.0")
    return lines


def _write_split(
    items: list[tuple[Path, list[str]]],
    split: str,
    dataset_dir: Path,
) -> None:
    """Copia imagenes y escribe sus etiquetas para un split (train/val)."""
    img_out = dataset_dir / "images" / split
    lbl_out = dataset_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, lines) in enumerate(items):
        stem = f"{img_path.parent.name}_{idx:05d}"
        dst_img = img_out / f"{stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_img)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(dataset_dir: Path = config.DATASET_DIR) -> Path:
    """Escribe el archivo YAML de configuracion del dataset para Ultralytics."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(config.CLASS_NAMES)},
    }
    config.DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DATASET_YAML, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return config.DATASET_YAML


def build_dataset(
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    raw_dir: Path = config.RAW_IMAGES_DIR,
    dataset_dir: Path = config.DATASET_DIR,
    use_autolabel: bool = True,
) -> dict:
    """Construye el dataset YOLO completo a partir de las imagenes crudas."""
    random.seed(cfg.seed)

    logger.info("Construccion de dataset iniciada (raw=%s)", raw_dir)
    species_images = _iter_species_images(raw_dir)
    if not species_images:
        logger.error("No se encontraron imagenes en %s para construir el dataset", raw_dir)
        raise RuntimeError(
            f"No se encontraron imagenes en {raw_dir}. Ejecuta primero el scraper "
            "o genera datos de ejemplo con 'sample'."
        )

    # Cargar modelo base para el auto-etiquetado (opcional).
    model = None
    if use_autolabel:
        try:
            from ultralytics import YOLO

            model = YOLO(config.BASE_MODEL)
            print(f"[dataset] Auto-etiquetado con modelo base '{config.BASE_MODEL}'.")
            logger.info("Auto-etiquetado habilitado con modelo base '%s'", config.BASE_MODEL)
        except Exception as exc:  # noqa: BLE001
            print(f"[dataset] No se pudo cargar el modelo base ({exc}). "
                  "Se usara etiquetado debil (imagen completa).")
            logger.warning(
                "No se pudo cargar el modelo base para auto-etiquetado (%s); "
                "se usara etiquetado debil", exc,
            )
    else:
        logger.info("Auto-etiquetado deshabilitado; se usara etiquetado por contenido")

    # Limpiar dataset previo.
    for sub in ("images", "labels"):
        shutil.rmtree(dataset_dir / sub, ignore_errors=True)

    train_items: list[tuple[Path, list[str]]] = []
    val_items: list[tuple[Path, list[str]]] = []

    for species_key, images in species_images.items():
        class_id = config.NAME_TO_ID[species_key]
        random.shuffle(images)
        n_val = max(1, int(len(images) * cfg.val_split)) if len(images) > 1 else 0
        val_set = set(images[:n_val])

        for img_path in images:
            lines = _autolabel_boxes(model, img_path, class_id, cfg.autolabel_conf)
            if img_path in val_set:
                val_items.append((img_path, lines))
            else:
                train_items.append((img_path, lines))

    # Garantizar que val no quede vacio (Ultralytics lo requiere).
    if not val_items and train_items:
        val_items.append(train_items[-1])

    _write_split(train_items, "train", dataset_dir)
    _write_split(val_items, "val", dataset_dir)
    yaml_path = write_yaml(dataset_dir)

    summary = {
        "train": len(train_items),
        "val": len(val_items),
        "classes": len(config.CLASS_NAMES),
        "yaml": str(yaml_path),
    }
    print(
        f"[dataset] Dataset listo: {summary['train']} train / {summary['val']} val, "
        f"{summary['classes']} clases -> {yaml_path}"
    )
    logger.info(
        "Dataset construido: %d train / %d val, %d clases -> %s",
        summary["train"], summary["val"], summary["classes"], yaml_path,
    )
    return summary
