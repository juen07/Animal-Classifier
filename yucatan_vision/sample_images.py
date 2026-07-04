"""Generador de imagenes sinteticas de ejemplo.

Permite probar el pipeline completo (dataset -> entrenamiento -> inferencia)
sin depender de la red ni del scraping real. Cada especie recibe imagenes con
una forma y color caracteristicos, de modo que un modelo YOLO pequeno pueda
aprender a distinguirlas en pocas epocas.

NO sustituye al scraper real; es unicamente una ayuda para la demostracion.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw

from . import config


def _color_for(key: str) -> tuple[int, int, int]:
    """Color determinista y distinto para cada especie."""
    digest = hashlib.md5(key.encode()).digest()
    return (digest[0], digest[1], digest[2])


def _draw_shape(draw: ImageDraw.ImageDraw, shape_id: int, box, color) -> None:
    x0, y0, x1, y1 = box
    if shape_id % 4 == 0:
        draw.ellipse(box, fill=color, outline=(0, 0, 0), width=3)
    elif shape_id % 4 == 1:
        draw.rectangle(box, fill=color, outline=(0, 0, 0), width=3)
    elif shape_id % 4 == 2:
        draw.polygon(
            [(x0, y1), ((x0 + x1) / 2, y0), (x1, y1)],
            fill=color,
            outline=(0, 0, 0),
        )
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.polygon(
            [(cx, y0), (x1, cy), (cx, y1), (x0, cy)],
            fill=color,
            outline=(0, 0, 0),
        )


def generate(
    per_species: int = 12,
    raw_dir: Path = config.RAW_IMAGES_DIR,
    size: int = 320,
    seed: int = 0,
) -> dict[str, int]:
    """Genera ``per_species`` imagenes sinteticas por especie."""
    rng = random.Random(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for class_idx, species in enumerate(config.SPECIES):
        folder = raw_dir / species.key
        folder.mkdir(parents=True, exist_ok=True)
        color = _color_for(species.key)

        for i in range(per_species):
            bg = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255))
            img = Image.new("RGB", (size, size), bg)
            draw = ImageDraw.Draw(img)

            # Forma de tamano y posicion variables (no siempre centrada) para
            # que el modelo aprenda a localizar objetos, no solo a clasificar.
            obj = rng.randint(int(size * 0.35), int(size * 0.7))
            x0 = rng.randint(0, size - obj)
            y0 = rng.randint(0, size - obj)
            box = (x0, y0, x0 + obj, y0 + obj)
            _draw_shape(draw, class_idx, box, color)
            img.save(folder / f"{species.key}_{i:03d}.jpg", quality=90)

        summary[species.key] = per_species

    total = sum(summary.values())
    print(f"[sample] Generadas {total} imagenes sinteticas en {raw_dir}")
    return summary


def make_scene(
    species_keys: list[str],
    out_path: Path,
    size: int = 640,
    seed: int = 0,
) -> Path:
    """Crea una imagen de prueba con varias especies (multiobjeto)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(img)

    n = len(species_keys)
    cols = max(1, int(n ** 0.5 + 0.999))
    cell = size // cols

    for idx, key in enumerate(species_keys):
        class_idx = config.NAME_TO_ID.get(key, idx)
        color = _color_for(key)
        r, c = divmod(idx, cols)
        pad = cell // 6
        box = (
            c * cell + pad + rng.randint(0, pad),
            r * cell + pad + rng.randint(0, pad),
            (c + 1) * cell - pad,
            (r + 1) * cell - pad,
        )
        _draw_shape(draw, class_idx, box, color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)
    print(f"[sample] Escena multiobjeto guardada en {out_path}")
    return out_path
