"""Capa de acceso a la base de datos veterinaria (SQLite).

Almacena los registros medicos de cada especie: nivel de riesgo, danos hacia
caninos (toxicidad), sintomas y primeros auxilios. Es local y ligera, sin
necesidad de servidores externos.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config
from .vet_data import VET_RECORDS

logger = logging.getLogger(__name__)

# Mensaje obligatorio de la regla estricta.
NO_RECORD_MESSAGE = "No hay registro de toxicidad hacia caninos"


@dataclass
class VetRecord:
    """Registro veterinario asociado a una especie."""

    species_key: str
    common_name: str
    scientific_name: str
    risk_level: str
    toxicity: str
    symptoms: str
    first_aid: str

    @property
    def has_toxicity_info(self) -> bool:
        """True solo si el campo de danos (toxicity) tiene contenido real."""
        return bool(self.toxicity and self.toxicity.strip())


def connect(db_path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    """Abre una conexion a la base de datos con acceso por nombre de columna."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = config.DB_PATH, *, seed: bool = True) -> None:
    """Crea la tabla de especies y (opcionalmente) la puebla con datos semilla."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS especies (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                species_key     TEXT UNIQUE NOT NULL,
                common_name     TEXT NOT NULL,
                scientific_name TEXT,
                risk_level      TEXT,
                toxicity        TEXT,
                symptoms        TEXT,
                first_aid       TEXT
            )
            """
        )
        conn.commit()
        logger.info('Base de datos veterinaria inicializada en "%s"', Path(db_path))
        if seed:
            _seed(conn)


def _seed(conn: sqlite3.Connection) -> None:
    """Inserta o actualiza los registros semilla definidos en ``vet_data``."""
    conn.executemany(
        """
        INSERT INTO especies
            (species_key, common_name, scientific_name,
             risk_level, toxicity, symptoms, first_aid)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(species_key) DO UPDATE SET
            common_name     = excluded.common_name,
            scientific_name = excluded.scientific_name,
            risk_level      = excluded.risk_level,
            toxicity        = excluded.toxicity,
            symptoms        = excluded.symptoms,
            first_aid       = excluded.first_aid
        """,
        VET_RECORDS,
    )
    conn.commit()
    logger.info("Datos veterinarios semilla cargados (%d registros)", len(VET_RECORDS))


def get_record(
    species_key: str, db_path: Path | str = config.DB_PATH
) -> VetRecord | None:
    """Devuelve el registro de una especie o ``None`` si no existe."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM especies WHERE species_key = ?", (species_key,)
        ).fetchone()
    if row is None:
        return None
    return VetRecord(
        species_key=row["species_key"],
        common_name=row["common_name"],
        scientific_name=row["scientific_name"],
        risk_level=row["risk_level"] or "",
        toxicity=row["toxicity"] or "",
        symptoms=row["symptoms"] or "",
        first_aid=row["first_aid"] or "",
    )


def describe_toxicity(species_key: str, db_path: Path | str = config.DB_PATH) -> dict:
    """Resumen listo para el reporte, aplicando la regla estricta.

    Si la especie no existe o su campo de danos (toxicity) esta vacio, se
    devuelve el mensaje obligatorio ``NO_RECORD_MESSAGE``.
    """
    record = get_record(species_key, db_path)

    # Regla estricta: sin registro o sin campo de danos -> mensaje obligatorio.
    if record is None or not record.has_toxicity_info:
        # Nombre comun de respaldo desde el catalogo cuando falta el registro.
        from .config import KEY_TO_SPECIES

        catalog = KEY_TO_SPECIES.get(species_key)
        common = (
            record.common_name
            if record and record.common_name
            else (catalog.common_name if catalog else species_key)
        )
        risk = record.risk_level if record and record.risk_level else "DESCONOCIDO"
        if record is None:
            logger.warning(
                'Especie "%s" sin registro en la base de datos -> regla estricta: "%s"',
                species_key,
                NO_RECORD_MESSAGE,
            )
        else:
            logger.warning(
                'Especie "%s" con campo de danos vacio -> regla estricta: "%s"',
                species_key,
                NO_RECORD_MESSAGE,
            )
        return {
            "species_key": species_key,
            "common_name": common,
            "risk_level": risk,
            "message": NO_RECORD_MESSAGE,
            "has_info": False,
            "toxicity": "",
            "symptoms": "",
            "first_aid": "",
        }

    return {
        "species_key": record.species_key,
        "common_name": record.common_name,
        "risk_level": record.risk_level or "DESCONOCIDO",
        "message": record.toxicity,
        "has_info": True,
        "toxicity": record.toxicity,
        "symptoms": record.symptoms,
        "first_aid": record.first_aid,
    }


def all_records(db_path: Path | str = config.DB_PATH) -> list[VetRecord]:
    """Devuelve todos los registros de la base de datos."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM especies ORDER BY species_key").fetchall()
    return [
        VetRecord(
            species_key=r["species_key"],
            common_name=r["common_name"],
            scientific_name=r["scientific_name"],
            risk_level=r["risk_level"] or "",
            toxicity=r["toxicity"] or "",
            symptoms=r["symptoms"] or "",
            first_aid=r["first_aid"] or "",
        )
        for r in rows
    ]
