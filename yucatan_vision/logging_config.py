"""Configuracion centralizada de la bitacora de auditoria (logging).

Toda la actividad relevante del sistema (eventos clave, operaciones de
archivos, advertencias y errores) se registra automaticamente en la carpeta
``logs/`` usando el modulo estandar ``logging``.

Formato de cada entrada:

    2026-07-18 09:42:13 — [INFO] File "students.csv" loaded successfully

Los loggers de cada modulo se crean con ``logging.getLogger(__name__)``, por lo
que cuelgan del logger del paquete ``yucatan_vision`` y heredan sus manejadores.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from . import config

# Nombre del logger raiz del paquete (todos los modulos cuelgan de este).
PACKAGE_LOGGER = "yucatan_vision"

# Formato solicitado: fecha/hora — [NIVEL] mensaje
LOG_FORMAT = "%(asctime)s — [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: int = logging.INFO,
    to_console: bool = True,
    daily: bool = False,
) -> logging.Logger:
    """Configura la bitacora de auditoria (idempotente).

    Args:
        level: Nivel minimo a registrar.
        to_console: Si ``True``, tambien emite las entradas por consola.
        daily: Si ``True`` usa un archivo por dia (``logs/YYYY-MM-DD.log``);
            de lo contrario usa ``logs/app.log``.

    Returns:
        El logger del paquete ya configurado.
    """
    global _configured

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if daily:
        log_file: Path = config.LOGS_DIR / f"{date.today().isoformat()}.log"
    else:
        log_file = config.LOGS_DIR / "app.log"

    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    # Evita duplicar en el logger raiz global.
    logger.propagate = False

    if _configured:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        console = logging.StreamHandler()
        console.setLevel(max(level, logging.WARNING))  # consola solo avisos/errores
        console.setFormatter(formatter)
        logger.addHandler(console)

    _configured = True
    logger.info('Bitacora de auditoria inicializada -> "%s"', log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger de modulo colgado del paquete."""
    return logging.getLogger(name)
