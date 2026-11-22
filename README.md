# Conecta+ · Predicción de churn a 90 días y decisión de retención

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adrianfelipe812/prueba-conecta/blob/main/solucion.ipynb)

Solución a la prueba técnica de Científico de Datos Senior. Responde dos preguntas con datos al 31-dic-2025:
**¿a quién contactar en los próximos 90 días?** y **¿por qué se están yendo los clientes?**

| Resultado | Valor |
|---|---|
| Clientes puntuados | 1.558 (`predicciones.csv`) |
| Clientes a contactar | 522, con la regla de valor esperado por plan |
| Modelo | Ensamble calibrado: regresión logística L1 + Random Forest (50/50) |
| Validación fuera de tiempo (sep-2025) | AUC 0,75 · 44 % del valor máximo posible (19,1 M COP vs 6,8 M contactando a todos) |
| Hallazgo principal | La versión Android 5.4.0 (ago-2025) triplica el churn de sus usuarios |

## Historia para el negocio
`storytelling.html` cuenta los hallazgos en tres actos (problema → causa → qué hacer), con gráficos interactivos y un
simulador de la campaña. Descárgalo y ábrelo en el navegador (usa Plotly desde su CDN). Se regenera con
`python -m conecta.storytelling`.

## Cómo ejecutarlo (Google Colab)
1. Abre `solucion.ipynb` con el botón **Open in Colab**.
2. Sube `datos_conecta.zip` al panel de archivos de Colab (queda en `/content/datos_conecta.zip`).
   Alternativa: móntalo desde Google Drive y ajusta la ruta en la sección 0.
3. *Entorno de ejecución → Ejecutar todo* (~5 minutos). La primera celda clona este repositorio para tener el paquete
   `conecta/` e instala `requirements.txt` (solo `mlflow`; el resto viene en Colab).

El notebook descomprime el zip validando las rutas, construye las capas de datos, entrena y compara los modelos,
registra los experimentos en MLflow (local, `sqlite:///mlflow.db`, sin servidor) y escribe `predicciones.csv`.

**Los datos no están en el repositorio.** El notebook espera el archivo `datos_conecta.zip` entregado con la prueba.

## Organización del repositorio
```
├── solucion.ipynb          Notebook principal (partes 1–7 del enunciado + MLflow)
├── predicciones.csv        cliente_id, probabilidad, contactar (una fila por cliente de clientes_a_puntuar)
├── storytelling.html       Historia de negocio interactiva (3 actos + simulador)
├── MEMO.md                 Una página para la gerencia (no técnica)
├── BITACORA_IA.md          Uso de inteligencia artificial y tiempo invertido
├── DECISIONES.md           Registro de decisiones con alternativas descartadas
├── requirements.txt        Librerías que Colab no trae
├── conecta/                Código (paquete Python)
│   ├── config.py           Supuestos de negocio, fechas y rutas (cambiar aquí)
│   ├── bronze.py           Ingesta fiel de las fuentes + metadatos (SHA-256, fila, fecha)
│   ├── silver.py           Limpieza, tipado, deduplicación, unificación de tickets, NLP
│   ├── gold.py             Variables calculadas al corte (sin información del futuro)
│   ├── quality.py          Controles de calidad: bronze mide, silver verifica
│   ├── nlp.py              Sarcasmo, intención, frustración (reglas validadas)
│   ├── negocio.py          Regla de contacto y valor neto de la decisión
│   ├── modelado.py         15 modelos, validación temporal, calibración, ensamble, MLflow
│   ├── storytelling.py     Genera storytelling.html
│   ├── experimento_*.py    Anotaciones: cortes adicionales y frecuencia de actualización
│   ├── pipeline.py         Orquestador bronze → silver → gold → calidad
│   └── recursos/           Sets de validación del NLP (sin datos de clientes)
├── notebooks/              Detalle por fase: 01 calidad · 02 NLP · 03 modelos
└── docs/                   Catálogo de calidad, diccionario de variables, estado vs enunciado
```

## Enfoque en breve
- **Datos (arquitectura medallion).** Se documentaron y resolvieron 15 problemas: dos esquemas distintos en los tickets,
  zonas horarias mezcladas, facturas y eventos duplicados, ciudades y edades inválidas, y fugas de información
  (`estado_actual`, fechas de pago). El pipeline se detiene si algún control falla. Ver `docs/calidad_datos.md`.
- **Texto y logs.** NLP por reglas validado contra etiquetas manuales, frases nuevas y un modelo preentrenado: detecta
  sarcasmo (p. ej., un cliente que "agradece" el servicio mientras reclama un cobro doble repetido) e intención de cancelar. Los logs se convierten
  en señales de uso y fallos de la app.
- **Modelo.** Validación temporal (marzo → junio, marzo+junio → septiembre), probabilidades calibradas y selección por
  **valor neto de la decisión**, no por AUC.
- **Decisión.** Contactar si `p > 25.000 / (0,40 × 12 × precio del plan)`. Por política, no se vuelve a contactar a
  quien ya recibió una oferta de retención (costo documentado; reversible en `config.py`).

## Ejecución local (opcional)
```bash
py -V:3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m conecta.pipeline      # requiere los datos en raw/datos_conecta_extraido/
.venv/Scripts/jupyter-lab.exe
```

Bonus (opcional): no incluido.
