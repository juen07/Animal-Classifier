"""Configuracion central del proyecto.

Aqui se define el catalogo de especies endemicas de Yucatan que el sistema
aprende a detectar, junto con todas las rutas del proyecto y los
hiperparametros por defecto usados en el resto del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"

# Imagenes crudas descargadas por el scraper (una carpeta por especie).
RAW_IMAGES_DIR: Path = DATA_DIR / "raw"
# Dataset ya curado y estructurado en formato YOLO.
DATASET_DIR: Path = DATA_DIR / "dataset"
# Archivo YAML que describe el dataset para Ultralytics.
DATASET_YAML: Path = DATASET_DIR / "yucatan.yaml"

# Base de datos veterinaria local.
DB_PATH: Path = DATA_DIR / "veterinaria.db"

# Modelos y resultados.
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Modelo base preentrenado usado como punto de partida del fine-tuning
# y para el auto-etiquetado del dataset.
BASE_MODEL: str = "yolov8n.pt"
# Ruta donde se copia el mejor modelo entrenado.
TRAINED_MODEL: Path = MODELS_DIR / "yucatan_best.pt"


@dataclass(frozen=True)
class Species:
    """Metadatos de una especie del catalogo.

    Attributes:
        key: Identificador interno (nombre de clase para YOLO, sin espacios).
        common_name: Nombre comun en espanol.
        scientific_name: Nombre cientifico.
        query: Termino de busqueda usado por el scraper.
    """

    key: str
    common_name: str
    scientific_name: str
    query: str


# Catalogo de especies endemicas / representativas de la peninsula de Yucatan.
# Se incluye una mezcla intencional de especies peligrosas/toxicas para caninos
# y especies inofensivas para demostrar la regla estricta de la base de datos.
SPECIES: list[Species] = [
    Species(
        key="sapo_gigante",
        common_name="Sapo gigante",
        scientific_name="Rhinella marina",
        query="Rhinella marina sapo gigante",
    ),
    Species(
        key="nauyaca",
        common_name="Nauyaca",
        scientific_name="Bothrops asper",
        query="Bothrops asper nauyaca serpiente",
    ),
    Species(
        key="coralillo",
        common_name="Serpiente coralillo",
        scientific_name="Micrurus diastema",
        query="Micrurus coralillo serpiente Yucatan",
    ),
    Species(
        key="alacran",
        common_name="Alacran de Yucatan",
        scientific_name="Centruroides gracilis",
        query="Centruroides gracilis alacran",
    ),
    Species(
        key="tarantula",
        common_name="Tarantula rodilla roja",
        scientific_name="Brachypelma vagans",
        query="Brachypelma vagans tarantula",
    ),
    Species(
        key="iguana_negra",
        common_name="Iguana negra",
        scientific_name="Ctenosaura similis",
        query="Ctenosaura similis iguana negra",
    ),
    Species(
        key="boa",
        common_name="Boa (ochkan)",
        scientific_name="Boa constrictor",
        query="Boa constrictor serpiente",
    ),
    Species(
        key="venado_cola_blanca",
        common_name="Venado cola blanca",
        scientific_name="Odocoileus virginianus",
        query="Odocoileus virginianus venado cola blanca",
    ),
    Species(
        key="pavo_ocelado",
        common_name="Pavo ocelado",
        scientific_name="Meleagris ocellata",
        query="Meleagris ocellata pavo ocelado",
    ),
    Species(
        key="coati",
        common_name="Coati / tejon",
        scientific_name="Nasua narica",
        query="Nasua narica coati tejon",
    ),
]

# Listas y mapeos derivados usados por YOLO (el indice = id de clase).
CLASS_NAMES: list[str] = [s.key for s in SPECIES]
NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_NAME: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
KEY_TO_SPECIES: dict[str, Species] = {s.key: s for s in SPECIES}


@dataclass
class PipelineConfig:
    """Hiperparametros por defecto del pipeline (sobrescribibles por CLI)."""

    # Scraping
    images_per_species: int = 40
    scraper_timeout: int = 15
    min_image_side: int = 64  # descartar imagenes muy pequenas
    scraper_adult_filter: bool = True

    # Dataset
    val_split: float = 0.2
    autolabel_conf: float = 0.15  # umbral bajo para el auto-etiquetado

    # Entrenamiento (fine-tuning en una sola pasada -> pocas epocas por defecto)
    epochs: int = 30
    imgsz: int = 640
    batch: int = 16
    seed: int = 0

    # Inferencia
    conf: float = 0.25
    iou: float = 0.45
    topk: int = 3

    extra: dict = field(default_factory=dict)


DEFAULT_CONFIG = PipelineConfig()


def ensure_dirs() -> None:
    """Crea todas las carpetas de trabajo si no existen."""
    for path in (
        DATA_DIR,
        RAW_IMAGES_DIR,
        DATASET_DIR,
        MODELS_DIR,
        RUNS_DIR,
        OUTPUTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
