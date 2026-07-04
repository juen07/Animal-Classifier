"""Punto de entrada de la demo.

Ejecuta la interfaz de linea de comandos del sistema de vision de Yucatan.

Ejemplos:
    uv run python main.py info
    uv run python main.py pipeline --use-sample --epochs 10
    uv run python main.py detect ruta/a/imagen.jpg
"""

from yucatan_vision.cli import main

if __name__ == "__main__":
    main()
