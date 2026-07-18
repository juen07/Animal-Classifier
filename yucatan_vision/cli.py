"""Interfaz de linea de comandos del sistema de vision de Yucatan.

Subcomandos disponibles:

    initdb    Crea/puebla la base de datos veterinaria (SQLite).
    scrape    Descarga y cura imagenes por especie (web scraping).
    sample    Genera imagenes sinteticas de ejemplo (sin red) para la demo.
    dataset   Construye el dataset YOLO (auto-etiquetado + split + YAML).
    train     Fine-tuning del modelo de deteccion de objetos.
    detect    Analiza una imagen: Top-3, base de datos e imagen anotada.
    pipeline  Ejecuta todo el flujo de la demo de principio a fin.
    info      Muestra el catalogo de especies y sus registros veterinarios.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from . import config, database
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def _build_cfg(args: argparse.Namespace) -> config.PipelineConfig:
    """Construye la configuracion combinando defaults con flags del CLI."""
    cfg = config.DEFAULT_CONFIG
    overrides = {}
    for name in (
        "images_per_species",
        "epochs",
        "imgsz",
        "batch",
        "conf",
        "iou",
        "topk",
        "val_split",
        "seed",
    ):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Handlers de cada subcomando
# --------------------------------------------------------------------------- #
def cmd_initdb(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    database.init_db(seed=not args.no_seed)
    print(f"[initdb] Base de datos lista en {config.DB_PATH}")


def cmd_info(args: argparse.Namespace) -> None:
    database.init_db(seed=True)
    print("Catalogo de especies:")
    for s in config.SPECIES:
        rec = database.describe_toxicity(s.key)
        flag = "OK " if rec["has_info"] else "!! "
        print(f"  [{flag}] {s.key:<20} {s.common_name:<26} riesgo={rec['risk_level']}")
    print("\nLeyenda: '!!' -> aplica la regla estricta "
          f"('{database.NO_RECORD_MESSAGE}').")


def cmd_scrape(args: argparse.Namespace) -> None:
    from . import scraper

    config.ensure_dirs()
    cfg = _build_cfg(args)
    scraper.scrape_dataset(cfg=cfg)


def cmd_sample(args: argparse.Namespace) -> None:
    from . import sample_images

    config.ensure_dirs()
    sample_images.generate(per_species=args.per_species, seed=args.seed or 0)


def cmd_dataset(args: argparse.Namespace) -> None:
    from . import dataset

    config.ensure_dirs()
    cfg = _build_cfg(args)
    dataset.build_dataset(cfg=cfg, use_autolabel=not args.no_autolabel)


def cmd_train(args: argparse.Namespace) -> None:
    from . import train

    config.ensure_dirs()
    cfg = _build_cfg(args)
    train.train_model(cfg=cfg)


def cmd_detect(args: argparse.Namespace) -> None:
    from . import detect

    config.ensure_dirs()
    database.init_db(seed=True)
    cfg = _build_cfg(args)
    detect.run(
        args.image,
        cfg=cfg,
        model_path=args.model,
        out_path=args.output,
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Ejecuta la demo completa de principio a fin."""
    from . import dataset, detect, sample_images, train

    config.ensure_dirs()
    cfg = _build_cfg(args)

    print("\n===== 1/5  Base de datos veterinaria =====")
    database.init_db(seed=True)

    print("\n===== 2/5  Recoleccion de imagenes =====")
    if args.use_sample:
        sample_images.generate(per_species=args.per_species, seed=cfg.seed)
    else:
        from . import scraper

        scraper.scrape_dataset(cfg=cfg)

    print("\n===== 3/5  Construccion del dataset =====")
    dataset.build_dataset(cfg=cfg, use_autolabel=not args.no_autolabel)

    print("\n===== 4/5  Entrenamiento (fine-tuning) =====")
    model_path = train.train_model(cfg=cfg)

    print("\n===== 5/5  Inferencia de demostracion =====")
    image = args.image
    if image is None:
        # Generar una escena multiobjeto de prueba.
        keys = [s.key for s in config.SPECIES[:4]]
        image = sample_images.make_scene(
            keys, config.OUTPUTS_DIR / "escena_demo.jpg", seed=cfg.seed
        )
    detect.run(image, cfg=cfg, model_path=model_path)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yucatan-vision",
        description="Deteccion de animales endemicos de Yucatan y evaluacion de "
        "toxicidad para caninos.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # initdb
    sp = sub.add_parser("initdb", help="Crear/poblar la base de datos veterinaria.")
    sp.add_argument("--no-seed", action="store_true", help="No insertar datos semilla.")
    sp.set_defaults(func=cmd_initdb)

    # info
    sp = sub.add_parser("info", help="Mostrar catalogo de especies y registros.")
    sp.set_defaults(func=cmd_info)

    # scrape
    sp = sub.add_parser("scrape", help="Descargar y curar imagenes (web scraping).")
    sp.add_argument("--images-per-species", type=int, dest="images_per_species")
    sp.set_defaults(func=cmd_scrape)

    # sample
    sp = sub.add_parser("sample", help="Generar imagenes sinteticas de ejemplo.")
    sp.add_argument("--per-species", type=int, default=12)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_sample)

    # dataset
    sp = sub.add_parser("dataset", help="Construir el dataset YOLO.")
    sp.add_argument("--val-split", type=float, dest="val_split")
    sp.add_argument("--no-autolabel", action="store_true",
                    help="Usar etiquetado debil (imagen completa) en vez de YOLO.")
    sp.add_argument("--seed", type=int, dest="seed")
    sp.set_defaults(func=cmd_dataset)

    # train
    sp = sub.add_parser("train", help="Fine-tuning del modelo YOLO.")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--imgsz", type=int)
    sp.add_argument("--batch", type=int)
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_train)

    # detect
    sp = sub.add_parser("detect", help="Analizar una imagen.")
    sp.add_argument("image", help="Ruta de la imagen a analizar.")
    sp.add_argument("--model", default=None, help="Ruta a un modelo .pt especifico.")
    sp.add_argument("--output", default=None, help="Ruta de la imagen anotada.")
    sp.add_argument("--conf", type=float)
    sp.add_argument("--iou", type=float)
    sp.add_argument("--topk", type=int)
    sp.set_defaults(func=cmd_detect)

    # pipeline
    sp = sub.add_parser("pipeline", help="Ejecutar la demo completa de extremo a extremo.")
    sp.add_argument("--image", default=None,
                    help="Imagen final a analizar (si se omite, se genera una escena).")
    sp.add_argument("--use-sample", action="store_true",
                    help="Usar imagenes sinteticas en vez del scraper real.")
    sp.add_argument("--per-species", type=int, default=12)
    sp.add_argument("--images-per-species", type=int, dest="images_per_species")
    sp.add_argument("--no-autolabel", action="store_true")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--imgsz", type=int)
    sp.add_argument("--batch", type=int)
    sp.add_argument("--conf", type=float)
    sp.add_argument("--iou", type=float)
    sp.add_argument("--topk", type=int)
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_pipeline)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Inicializa la bitacora de auditoria antes de ejecutar cualquier comando.
    config.ensure_dirs()
    setup_logging()
    logger.info("Ejecutando comando '%s'", args.command)

    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - se registra y se re-lanza
        logger.error("El comando '%s' fallo: %s", args.command, exc, exc_info=True)
        logger.info("Comando '%s' finalizado con errores", args.command)
        raise
    else:
        logger.info("Comando '%s' finalizado correctamente", args.command)


if __name__ == "__main__":
    main()
