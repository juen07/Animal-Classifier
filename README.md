===========================================================================
===========================================================================

## Instrucciones y Objetivos del Proyecto

*   **Propósito Principal:** Desarrollar un sistema de visión artificial capaz de recibir la ruta de una imagen por parte del usuario, detectar múltiples animales endémicos de Yucatán de forma simultánea y evaluar su nivel de toxicidad o peligro para mascotas caninas.
*   **Recolección Automática de Datos:** Utilizar técnicas de web scraping para descargar imágenes de las especies locales y construir el dataset de entrenamiento automáticamente, purgando los archivos corruptos o no válidos.
*   **Entrenamiento del Modelo (Fine-Tuning):** Entrenar un modelo de detección de objetos en una sola pasada que aprenda a ubicar e identificar las especies específicas a partir del dataset recolectado.
*   **Evaluación Multiobjeto:** Asegurar que el algoritmo no solo clasifique la imagen de manera global, sino que ubique espacialmente a cada animal presente en la foto mediante cuadros delimitadores.
*   **Integración Relacional:** Conectar las predicciones generadas por la IA con una base de datos local que almacene información médica veterinaria (Nivel de riesgo, Síntomas y Primeros auxilios).
*   **Manejo de Excepciones (Regla Estricta):** Implementar una validación de seguridad donde, si un animal detectado no cuenta con información o tiene su campo de daños vacío en la base de datos, el sistema emita obligatoriamente el mensaje: "No hay registro de toxicidad hacia caninos".
*   **Generación de Resultados:** Devolver un reporte textual en consola con el Top 3 de especies más probables por cada detección, y exportar la imagen original con los cuadros delimitadores y niveles de riesgo dibujados sobre cada animal.

---

## Stack Tecnológico

*   **Lenguaje de Programación:** Python.
*   **Inteligencia Artificial (Detección de Objetos):** YOLO (mediante la librería oficial de `ultralytics`) para el entrenamiento y la inferencia espacial.
*   **Web Scraping y Curación de Datos:** `bing-image-downloader` para la descarga masiva de imágenes y `Pillow` (PIL) para la validación y filtrado de archivos corruptos.
*   **Almacenamiento y Base de Datos:** SQLite (`sqlite3`) para gestionar los registros veterinarios de forma local y ligera sin necesidad de servidores externos.
*   **Procesamiento Visual:** Herramientas nativas de Ultralytics u OpenCV para el renderizado final de los recuadros de alerta y la exportación de la imagen analizada.

===========================================================================
[FIN DEL ARCHIVO]
===========================================================================
