"""Datos veterinarios semilla para la base de datos local.

Cada registro describe el riesgo que representa una especie para un canino.

NOTA IMPORTANTE sobre la "regla estricta":

    El campo ``toxicity`` (danos hacia caninos) es el campo critico. Si una
    especie detectada NO existe en la base de datos, o existe pero su campo
    ``toxicity`` esta vacio, el sistema debe emitir obligatoriamente el mensaje:

        "No hay registro de toxicidad hacia caninos"

De forma intencional dejamos:
    * ``coati``        -> presente pero con ``toxicity`` vacio (campo de danos vacio).
    * ``pavo_ocelado`` -> ausente por completo de la base de datos.

Ambos casos deben disparar la regla estricta durante la inferencia.
"""

from __future__ import annotations

# Cada tupla: (species_key, common_name, scientific_name,
#              risk_level, toxicity, symptoms, first_aid)
VET_RECORDS: list[tuple[str, str, str, str, str, str, str]] = [
    (
        "sapo_gigante",
        "Sapo gigante",
        "Rhinella marina",
        "ALTO",
        "Secrecion de bufotoxinas por las glandulas parotidas; altamente "
        "toxico si el perro lo lame o muerde.",
        "Salivacion excesiva, encias rojas, vomito, temblores, convulsiones y "
        "arritmias cardiacas.",
        "Enjuagar la boca con agua corriente de adelante hacia atras durante "
        "10 minutos (evitar tragar) y acudir de inmediato al veterinario.",
    ),
    (
        "nauyaca",
        "Nauyaca",
        "Bothrops asper",
        "ALTO",
        "Veneno hemotoxico por mordedura; provoca necrosis y hemorragias.",
        "Dolor intenso, inflamacion rapida, sangrado, moretones y debilidad.",
        "Inmovilizar la zona, mantener al perro en reposo por debajo del "
        "nivel del corazon y trasladar urgentemente para antiveneno.",
    ),
    (
        "coralillo",
        "Serpiente coralillo",
        "Micrurus diastema",
        "ALTO",
        "Veneno neurotoxico por mordedura; afecta el sistema nervioso.",
        "Paralisis progresiva, dificultad respiratoria, salivacion y pupilas "
        "dilatadas.",
        "Evitar manipular la herida, mantener la calma del animal y acudir de "
        "urgencia; se requiere antiveneno especifico.",
    ),
    (
        "alacran",
        "Alacran de Yucatan",
        "Centruroides gracilis",
        "MEDIO",
        "Veneno neurotoxico por picadura; usualmente doloroso, rara vez letal "
        "en perros sanos.",
        "Dolor local, inflamacion, temblores, inquietud y salivacion.",
        "Aplicar frio local, vigilar signos vitales y consultar al "
        "veterinario si hay temblores o dificultad para respirar.",
    ),
    (
        "tarantula",
        "Tarantula rodilla roja",
        "Brachypelma vagans",
        "BAJO",
        "Pelos urticantes y mordedura leve; irritacion mas que toxicidad "
        "sistemica.",
        "Estornudos, irritacion ocular o bucal, salivacion y molestia local.",
        "Lavar la zona con agua, evitar que el perro se frote los ojos y "
        "consultar si hay irritacion persistente.",
    ),
    (
        "boa",
        "Boa (ochkan)",
        "Boa constrictor",
        "BAJO",
        "No venenosa; el riesgo es por constriccion o mordedura mecanica.",
        "Heridas por mordida, estres y posibles marcas de presion.",
        "Separar con cuidado a los animales, limpiar y desinfectar heridas y "
        "vigilar signos de dolor.",
    ),
    (
        "iguana_negra",
        "Iguana negra",
        "Ctenosaura similis",
        "BAJO",
        "No venenosa; riesgo por mordida defensiva o latigazos con la cola.",
        "Heridas superficiales por mordida o arananzos.",
        "Limpiar y desinfectar la herida; vigilar signos de infeccion.",
    ),
    (
        "venado_cola_blanca",
        "Venado cola blanca",
        "Odocoileus virginianus",
        "NINGUNO",
        "Especie herbivora no toxica; sin riesgo quimico para el canino.",
        "Sin sintomas de toxicidad esperados.",
        "No requiere primeros auxilios por toxicidad.",
    ),
    # --- Casos de la regla estricta ------------------------------------- #
    (
        # Campo de danos (toxicity) intencionalmente vacio.
        "coati",
        "Coati / tejon",
        "Nasua narica",
        "",
        "",  # <- campo de danos vacio -> debe disparar la regla estricta
        "",
        "",
    ),
    # 'pavo_ocelado' se omite a proposito para simular una especie sin registro.
]
