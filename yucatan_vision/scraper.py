"""Recoleccion automatica de datos (web scraping) y curacion de imagenes.

Usa ``bing-image-downloader`` para descargar imagenes de cada especie y
``Pillow`` para validar y purgar los archivos corruptos, demasiado pequenos o
que no sean imagenes reales.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import config
from .config import Species

logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _download_species(
    species: Species,
    limit: int,
    out_dir: Path,
    timeout: int,
    adult_filter: bool,
) -> Path:
    """Descarga imagenes de una especie usando bing-image-downloader.

    ``bing_image_downloader`` crea una subcarpeta con el nombre de la query.
    La renombramos a ``species.key`` para tener nombres de clase limpios.
    """
    from bing_image_downloader import downloader

    out_dir.mkdir(parents=True, exist_ok=True)

    downloader.download(
        species.query,
        limit=limit,
        output_dir=str(out_dir),
        adult_filter_off=not adult_filter,
        force_replace=False,
        timeout=timeout,
        verbose=False,
    )

    downloaded_dir = out_dir / species.query
    target_dir = out_dir / species.key

    if downloaded_dir.exists() and downloaded_dir != target_dir:
        if target_dir.exists():
            # Mover archivos nuevos al directorio destino existente.
            for f in downloaded_dir.iterdir():
                shutil.move(str(f), str(target_dir / f.name))
            shutil.rmtree(downloaded_dir, ignore_errors=True)
        else:
            downloaded_dir.rename(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def validate_and_purge(directory: Path, min_side: int) -> tuple[int, int]:
    """Valida imagenes con Pillow y elimina las corruptas o invalidas.

    Devuelve una tupla ``(validas, purgadas)``.
    """
    valid = 0
    purged = 0
    if not directory.exists():
        return (0, 0)

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            path.unlink(missing_ok=True)
            purged += 1
            continue
        try:
            # Primera pasada: verifica integridad del archivo.
            with Image.open(path) as img:
                img.verify()
            # Segunda pasada: reabrir para leer dimensiones reales
            # (verify() invalida el objeto imagen).
            with Image.open(path) as img:
                width, height = img.size
                if min(width, height) < min_side:
                    raise ValueError("imagen demasiado pequena")
                # Normaliza a RGB para descartar modos raros / corruptos.
                img.convert("RGB")
            valid += 1
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            path.unlink(missing_ok=True)
            purged += 1
            logger.debug('Purgada imagen invalida "%s": %s', path.name, exc)

    return (valid, purged)


def scrape_dataset(
    species_list: list[Species] | None = None,
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    raw_dir: Path = config.RAW_IMAGES_DIR,
) -> dict[str, int]:
    """Descarga y cura imagenes para todas las especies.

    Devuelve un diccionario ``{species_key: numero_de_imagenes_validas}``.
    """
    species_list = species_list or config.SPECIES
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    logger.info(
        "Inicio de web scraping: %d especies, %d imagenes objetivo por especie",
        len(species_list),
        cfg.images_per_species,
    )

    for species in species_list:
        print(f"[scraper] Descargando '{species.common_name}' ({species.query}) ...")
        logger.info(
            'Descargando especie "%s" (query="%s")', species.key, species.query
        )
        try:
            target = _download_species(
                species,
                limit=cfg.images_per_species,
                out_dir=raw_dir,
                timeout=cfg.scraper_timeout,
                adult_filter=cfg.scraper_adult_filter,
            )
        except Exception as exc:  # noqa: BLE001 - la red puede fallar
            print(f"[scraper]   ! Error al descargar {species.key}: {exc}")
            logger.error('Fallo la descarga de "%s": %s', species.key, exc)
            summary[species.key] = 0
            continue

        valid, purged = validate_and_purge(target, cfg.min_image_side)
        print(f"[scraper]   -> {valid} validas, {purged} purgadas")
        logger.info(
            'Especie "%s": %d imagenes validas, %d purgadas (corruptas/invalidas)',
            species.key,
            valid,
            purged,
        )
        if valid == 0:
            logger.warning('La especie "%s" no obtuvo imagenes validas', species.key)
        summary[species.key] = valid

    total = sum(summary.values())
    print(f"[scraper] Total de imagenes validas: {total}")
    logger.info("Web scraping finalizado: %d imagenes validas en total", total)
    return summary
