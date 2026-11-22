"""Pipeline medallion para la prueba Conecta+ (churn 90 días).

Capas:
    bronze  -> copia fiel de las fuentes + metadatos de ingesta (sin limpiar)
    silver  -> entidades limpias, tipadas, deduplicadas y conformadas
    gold    -> tablas listas para modelar (spine cliente x fecha_corte, features)

Uso:
    from conecta import pipeline
    pipeline.run(data_dir="raw/datos_conecta_extraido", lake_dir="lake")
"""
