# Calidad de datos — catálogo de problemas

Generado en la fase 1 (2026-11-19). Evidencia ejecutable: `notebooks/01_calidad_datos.ipynb`.
Reporte automático: `lake/_calidad/reporte_calidad.csv` (37 checks, 0 fallas en silver).

## Arquitectura medallion

```
raw/ (solo lectura, verificado con SHA-256)
  └─ bronze  copia fiel + metadatos de ingesta  ──  conecta/bronze.py
      └─ silver  limpio, tipado, deduplicado, hora local Bogotá  ──  conecta/silver.py
          └─ gold  spine cliente × fecha_corte (+ features en la fase 2)  ──  conecta/gold.py
quality.py: mide problemas en bronze y verifica en silver · pipeline.py: orquesta todo
```
Ejecutar: `.venv/Scripts/python.exe -m conecta.pipeline` (falla si algún check de silver falla).

## Problemas encontrados

| Código | Tabla | Problema | Magnitud | Tratamiento (silver) |
|---|---|---|---|---|
| P01 | clientes | Ciudad con 17 variantes de 6 ciudades (`B/quilla`, `BOGOTA`, `bogota d.c.`) | 2.000 filas (67 %) | Mapa a nombre canónico; se conserva `ciudad_original` |
| P02 | clientes | Edades centinela 0 / 99 / 999 | 85 (2,8 %) | NaN + `flag_edad_invalida`; imputar en gold |
| P03 | clientes | Contacto de retención anterior a la fecha de alta | 76 (2,5 %) | `flag_contacto_antes_alta` |
| P04 | clientes | `estado_actual` = foto de hoy → **fuga** | 100 % | Se conserva solo para validar; prohibido como feature |
| P05 | suscripciones | Cierre por `cambio_plan` no es churn | 372 subs | `fin_es_churn` / `fin_es_cambio_plan`; regla reproduce 4.810/4.810 etiquetas |
| P06 | facturas | `factura_id` duplicado por recarga (día 5, 07:00), contenido idéntico | 779 (2,2 %) | Deduplicar conservando la primera carga; `n_cargas` |
| P07 | facturas | `fecha_pago` se actualiza sin fecha de carga propia → **fuga** si no se corta | 1.457 sin pago | En gold: pago conocido solo si `fecha_pago <= corte` |
| P08 | facturas | Básico cobrado a 19.900 hasta mar-2025 y 23.900 desde abr-2025 | 5.784 | No es error: subida de precio del 20 % (hallazgo) |
| P09 | tickets | Dos esquemas JSON: A (hasta 2025-06-10, inglés) y B (desde 2025-06-10, español, anidado) | 2.468 B / 2.210 A | Unificación a un solo esquema (`esquema` indica origen) |
| P10 | tickets | ID de cliente `"CL-010792"` (A) vs entero (B, SQL) | 2.210 | Convertir a entero; 100 % de match con clientes |
| P11 | tickets | csat ausente (abiertos + 22 % de resueltos B) | 999 | Mantener nulo, no imputar |
| P12 | tickets | Zonas horarias mezcladas (UTC `Z` vs `-05:00`) | 2.468 | Todo a hora local Bogotá; evita 6 tickets "de 2026" |
| P13 | eventos | Eventos duplicados exactos | 1.463 (1,4 %) | Eliminados |
| P15 | tickets | Sarcasmo: quejas con palabras positivas (p. ej., un cliente que "agradece" el servicio mientras reclama un cobro doble repetido) | 974 tickets (20,8 %), 9 de 38 textos | NLP por reglas en silver: `nlp_sarcasmo`, `nlp_sentimiento`, `nlp_intencion`, `nlp_frustracion`, `nlp_recurrente`, `nlp_tema_precio` |

## NLP de tickets (P15) — resumen de validación
Detalle: `notebooks/02_nlp_tickets.ipynb` · código: `conecta/nlp.py`, `conecta/nlp_validacion.py`.

| Set | Sarcasmo (exactitud / F1) | Sentimiento F1 macro | Intención exactitud |
|---|---|---|---|
| Referencia: 38 textos reales (etiqueta manual) | 1,00 / 1,00 | 0,86 | 1,00 |
| Desarrollo: 30 frases nuevas (usado para ajustar) | 1,00 / 1,00 | 1,00 | 1,00 |
| **Prueba: 24 frases nuevas nunca usadas** | **0,79 / 0,55** | **0,81** | **0,96** |
| pysentimiento en prueba (modelo externo) | 0,67 / 0,43 | 0,75 | — |
| Léxico ingenuo (sin sarcasmo) en prueba | — | 0,54 | — |

- En los datos reales, las reglas son exactas en sarcasmo e intención. Con textos nuevos generalizan peor: hay que ampliar el léxico y monitorear los textos que caen fuera del set de referencia (check P15).
- pysentimiento confirma que hay ironía, pero marca como "positivas" las quejas sarcásticas y rinde peor que las reglas.
- Relación con el churn: lo más predictivo es la intención (cancelar → 100 %, amenaza → 21 %, bajar de plan → 8 %) y la frustración alta (40 % vs 30 % base). El sarcasmo suma de 1 a 6 pp dentro de cada categoría. El sentimiento solo engaña: las cancelaciones corteses son "neutras".

## Anotación: cortes de entrenamiento adicionales
Experimento (`conecta/experimento_cortes.py`, prueba fuera de tiempo en 2025-09-30): los cortes mensuales de 2025 suben el AUC +0,004 a +0,010, con IC 95 % que incluye 0. Agregar 2023–2024 lo baja de 0,011 a 0,017 y descalibra el modelo. **No es relevante: se usan los 3 cortes oficiales.**

## Señales para el hallazgo abierto
1. Churn creciente: 10,3 % → 13,8 % → 15,7 % (cortes mar/jun/sep-2025).
2. Subida de precio del Básico (abr-2025) + tickets que reclaman el aumento sin aviso previo.
3. App Android 5.4.0 (1-ago-2025): `CRASH_ON_START` pasa de ~15 a ~2.000/mes; nace la categoría de ticket `app_se_cierra` (~1.070 tickets). Textos con sarcasmo.
4. Retención reactiva: el contacto llega 3–19 días antes de cancelar y el 94 % de contactados cancela igual.
5. Facturas sin pago en aumento desde may-2025.
