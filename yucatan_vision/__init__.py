"""Sistema de vision artificial para animales endemicos de Yucatan.

Este paquete implementa un pipeline completo de deteccion de objetos:

1. ``scraper``  -> Descarga y curacion automatica de imagenes (web scraping).
2. ``dataset``  -> Construccion del dataset en formato YOLO (auto-etiquetado).
3. ``train``    -> Fine-tuning de un modelo YOLO de deteccion de objetos.
4. ``database`` -> Registros veterinarios en SQLite (riesgo, sintomas, primeros auxilios).
5. ``detect``   -> Inferencia multiobjeto, reporte Top-3 e imagen anotada.
"""

from __future__ import annotations

__version__ = "0.1.0"
