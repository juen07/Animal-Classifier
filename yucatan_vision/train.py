"""Entrenamiento (fine-tuning) del modelo de deteccion de objetos.

Parte de un modelo YOLO preentrenado y lo ajusta en una sola pasada de
entrenamiento sobre el dataset de especies de Yucatan. El mejor peso resultante
se copia a ``config.TRAINED_MODEL`` para su uso en inferencia.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

from . import config

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Elige GPU si esta disponible, si no CPU."""
    return "0" if torch.cuda.is_available() else "cpu"


def train_model(
    cfg: config.PipelineConfig = config.DEFAULT_CONFIG,
    data_yaml: Path = config.DATASET_YAML,
    base_model: str = config.BASE_MODEL,
    run_name: str = "yucatan_finetune",
) -> Path:
    """Ejecuta el fine-tuning y devuelve la ruta del mejor modelo entrenado."""
    from ultralytics import YOLO

    if not Path(data_yaml).exists():
        logger.error("No existe el YAML del dataset: %s", data_yaml)
        raise FileNotFoundError(
            f"No existe el YAML del dataset: {data_yaml}. Construye el dataset primero."
        )

    device = _select_device()
    print(f"[train] Fine-tuning de '{base_model}' | device={device} | "
          f"epochs={cfg.epochs} | imgsz={cfg.imgsz}")
    logger.info(
        "Inicio de fine-tuning: modelo base='%s', device=%s, epochs=%d, imgsz=%d, batch=%d",
        base_model, device, cfg.epochs, cfg.imgsz, cfg.batch,
    )

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        device=device,
        project=str(config.RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
        plots=False,
    )

    # Localizar el mejor peso generado por Ultralytics.
    save_dir = Path(getattr(results, "save_dir", config.RUNS_DIR / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():
        logger.error("No se encontro el peso entrenado en %s", save_dir)
        raise FileNotFoundError(f"No se encontro el peso entrenado en {save_dir}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.TRAINED_MODEL)
    print(f"[train] Modelo entrenado guardado en {config.TRAINED_MODEL}")
    logger.info(
        "Fine-tuning finalizado. Mejor modelo copiado a '%s'", config.TRAINED_MODEL
    )
    return config.TRAINED_MODEL
