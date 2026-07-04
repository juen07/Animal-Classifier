# Visión Artificial — Animales Endémicos de Yucatán

Sistema de visión por computadora que recibe la **ruta de una imagen**, detecta
de forma **simultánea** varios animales endémicos de Yucatán mediante cuadros
delimitadores y **evalúa su nivel de toxicidad o peligro para mascotas
caninas** consultando una base de datos veterinaria local.

El proyecto es una **demo completa de extremo a extremo**: recolección
automática de datos (web scraping), curación de imágenes, construcción del
dataset, fine-tuning de un detector YOLO, inferencia multiobjeto y generación
de un reporte con imagen anotada.

---

## Tabla de contenido

- [Stack tecnológico](#stack-tecnológico)
- [Requisitos](#requisitos)
- [Instalación con UV](#instalación-con-uv)
- [Uso rápido (demo)](#uso-rápido-demo)
- [Comandos disponibles](#comandos-disponibles)
- [Flujo real con web scraping](#flujo-real-con-web-scraping)
- [Arquitectura del proyecto](#arquitectura-del-proyecto)
- [La regla estricta de toxicidad](#la-regla-estricta-de-toxicidad)
- [Especificación original del proyecto](#especificación-original-del-proyecto)

---

## Stack tecnológico

| Área | Herramienta |
| --- | --- |
| Lenguaje | Python 3.11 |
| Gestión de entorno | **UV** |
| Detección de objetos | `ultralytics` (YOLO) |
| Web scraping | `bing-image-downloader` |
| Curación de imágenes | `Pillow` (PIL) |
| Base de datos | `sqlite3` (librería estándar) |
| Procesamiento visual | Ultralytics / OpenCV / Pillow |

---

## Requisitos

- [UV](https://docs.astral.sh/uv/) instalado.
- Conexión a internet la **primera** vez (para descargar el modelo base
  `yolov8n.pt` y, si se usa, para el web scraping).

Instalar UV (si aún no lo tienes):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Instalación con UV

Todas las dependencias se gestionan con UV. Basta con sincronizar el entorno:

```bash
uv sync
```

> Cualquier librería adicional debe añadirse con `uv add <paquete>` para no
> contaminar la instalación global de Python.

---

## Uso rápido (demo)

La forma más sencilla de ver todo el sistema funcionando **sin depender de la
red** es usar imágenes sintéticas de ejemplo:

```bash
uv run python main.py pipeline --use-sample --per-species 18 --epochs 20
```

Esto ejecuta las 5 etapas:

1. Crea/puebla la base de datos veterinaria (SQLite).
2. Genera imágenes de ejemplo por especie.
3. Construye el dataset YOLO (auto-etiquetado + split + YAML).
4. Hace fine-tuning del modelo YOLO.
5. Analiza una escena de prueba: imprime el **Top-3** por detección, consulta
   la base de datos y exporta la imagen anotada en `outputs/`.

Para analizar una imagen propia con el modelo ya entrenado:

```bash
uv run python main.py detect ruta/a/mi_imagen.jpg
```

El reporte se imprime en consola y la imagen anotada se guarda en
`outputs/mi_imagen_analizado.jpg`.

Ejemplo de salida en consola:

```
[Deteccion #1] Sapo gigante  (riesgo: ALTO)
  Ubicacion (bbox): (74.2, 411.8) - (267.8, 589.5)
  Top 3 especies mas probables:
    1. Sapo gigante                  98.1%
    2. Nauyaca                        1.2%
    3. Boa (ochkan)                   0.4%
  Toxicidad/danos: Secrecion de bufotoxinas por las glandulas parotidas...
  Sintomas: Salivacion excesiva, encias rojas, vomito, temblores...
  Primeros auxilios: Enjuagar la boca con agua corriente...

[Deteccion #2] Coati / tejon  (riesgo: DESCONOCIDO)
  ...
  No hay registro de toxicidad hacia caninos
```

---

## Comandos disponibles

```bash
uv run python main.py <comando> [opciones]
```

| Comando | Descripción |
| --- | --- |
| `initdb` | Crea y puebla la base de datos veterinaria. |
| `info` | Muestra el catálogo de especies y sus registros. |
| `scrape` | Descarga y cura imágenes reales (web scraping). |
| `sample` | Genera imágenes sintéticas de ejemplo (offline). |
| `dataset` | Construye el dataset YOLO (auto-etiquetado + split). |
| `train` | Fine-tuning del modelo YOLO. |
| `detect <img>` | Analiza una imagen y exporta el resultado. |
| `pipeline` | Ejecuta toda la demo de extremo a extremo. |

Opciones útiles: `--epochs`, `--imgsz`, `--batch`, `--conf`, `--iou`,
`--topk`, `--images-per-species`. Consulta la ayuda de cada comando:

```bash
uv run python main.py detect --help
```

---

## Flujo real con web scraping

Para construir el dataset a partir de imágenes reales descargadas de internet:

```bash
uv run python main.py initdb
uv run python main.py scrape --images-per-species 60
uv run python main.py dataset
uv run python main.py train --epochs 60
uv run python main.py detect ruta/a/mi_imagen.jpg
```

- **`scrape`** descarga imágenes con `bing-image-downloader` y **purga** los
  archivos corruptos, no válidos o demasiado pequeños con `Pillow`.
- **`dataset`** realiza un **auto-etiquetado**: usa un YOLO preentrenado para
  ubicar el objeto principal de cada imagen; si no encuentra nada, recorta la
  región saliente (no-fondo) o, en última instancia, usa la imagen completa.

> Nota: el auto-etiquetado es una aproximación práctica para una demo. Para
> producción se recomienda revisar/corregir las anotaciones manualmente.

---

## Arquitectura del proyecto

```
yucatan_vision/
├── config.py         # Catálogo de especies, rutas e hiperparámetros
├── vet_data.py       # Datos veterinarios semilla
├── database.py       # Capa SQLite + regla estricta de toxicidad
├── scraper.py        # Web scraping + curación con Pillow
├── sample_images.py  # Generador de imágenes sintéticas (demo offline)
├── dataset.py        # Auto-etiquetado + dataset YOLO + YAML
├── train.py          # Fine-tuning del detector YOLO
├── detect.py         # Inferencia multiobjeto, Top-3, DB e imagen anotada
└── cli.py            # Interfaz de línea de comandos
main.py               # Punto de entrada
```

Artefactos generados (ignorados por git): `data/`, `models/`, `runs/`,
`outputs/`.

---

## La regla estricta de toxicidad

Si un animal detectado **no tiene registro** en la base de datos, o su campo de
daños (`toxicity`) está **vacío**, el sistema emite obligatoriamente:

> **"No hay registro de toxicidad hacia caninos"**

Para demostrarlo, la base de datos incluye a propósito dos casos límite:

- `coati` → existe pero con el campo de daños vacío.
- `pavo_ocelado` → ausente por completo de la base de datos.

Ambos disparan el mensaje obligatorio durante la inferencia.

---

## Especificación original del proyecto

<details>
<summary>Ver instrucciones y objetivos originales</summary>

### Instrucciones y Objetivos del Proyecto

- **Propósito Principal:** Desarrollar un sistema de visión artificial capaz de
  recibir la ruta de una imagen por parte del usuario, detectar múltiples
  animales endémicos de Yucatán de forma simultánea y evaluar su nivel de
  toxicidad o peligro para mascotas caninas.
- **Recolección Automática de Datos:** Utilizar técnicas de web scraping para
  descargar imágenes de las especies locales y construir el dataset de
  entrenamiento automáticamente, purgando los archivos corruptos o no válidos.
- **Entrenamiento del Modelo (Fine-Tuning):** Entrenar un modelo de detección
  de objetos en una sola pasada que aprenda a ubicar e identificar las especies
  específicas a partir del dataset recolectado.
- **Evaluación Multiobjeto:** Asegurar que el algoritmo no solo clasifique la
  imagen de manera global, sino que ubique espacialmente a cada animal presente
  en la foto mediante cuadros delimitadores.
- **Integración Relacional:** Conectar las predicciones generadas por la IA con
  una base de datos local que almacene información médica veterinaria (Nivel de
  riesgo, Síntomas y Primeros auxilios).
- **Manejo de Excepciones (Regla Estricta):** Implementar una validación de
  seguridad donde, si un animal detectado no cuenta con información o tiene su
  campo de daños vacío en la base de datos, el sistema emita obligatoriamente
  el mensaje: "No hay registro de toxicidad hacia caninos".
- **Generación de Resultados:** Devolver un reporte textual en consola con el
  Top 3 de especies más probables por cada detección, y exportar la imagen
  original con los cuadros delimitadores y niveles de riesgo dibujados sobre
  cada animal.

### Stack Tecnológico

- **Lenguaje:** Python.
- **IA (Detección de Objetos):** YOLO (`ultralytics`).
- **Web Scraping y Curación:** `bing-image-downloader` y `Pillow` (PIL).
- **Base de Datos:** SQLite (`sqlite3`).
- **Procesamiento Visual:** Ultralytics u OpenCV.

</details>
