# Estado del proyecto vs. enunciado

**Estado: terminado y entregable.** Última actualización: 2026-11-22, ~13:30.
**Tiempo invertido:** 3,9 horas (detalle por sesión en `BITACORA_IA.md`).
Fuente de requisitos: `raw/Enunciado - Prueba técnica Ciencia de Datos.pdf` (revisado completo).

Leyenda: ✅ hecho · ➖ opcional, no incluido

## Partes a resolver (Senior: 1–6 obligatorias)

| # | Parte (enunciado) | Estado | Dónde |
|---|---|---|---|
| 1 | Entender y preparar datos: explorar, limpiar, integrar SQL + JSON, documentar problemas y solución | ✅ | `solucion.ipynb` §1 · detalle en notebook 01 y `docs/calidad_datos.md` (15 problemas, 42 controles) |
| 2 | Modelo de churn a 90 días, justificar validación y métricas | ✅ | `solucion.ipynb` §3 · 15 modelos, validación temporal, calibración · detalle en notebook 03 |
| 3 | Decisión de negocio con los supuestos económicos + valor estimado | ✅ | `solucion.ipynb` §5 · regla de valor esperado por plan · `predicciones.csv` |
| 4 | Datos no estructurados (texto de tickets + logs): enfoque, justificación, límites | ✅ | `solucion.ipynb` §2 · NLP con sarcasmo validado · detalle en notebook 02 |
| 5 | Hallazgo abierto con evidencia + acción propuesta | ✅ | `solucion.ipynb` §6, `storytelling.html` y `MEMO.md`: la versión Android 5.4.0 triplica el churn (27 % vs 8 %; ~49 M COP/año en riesgo). Secundarios: retención reactiva y subida del Básico |
| 6 | Llevarlo a producción (media página) | ✅ | `solucion.ipynb` §7 · incluye la evidencia de frecuencia de actualización (reentrenar trimestral; puntuar mensual) |

## MLflow (extra)

| Requisito | Estado |
|---|---|
| Local, sin servidor ni cuentas | ✅ `sqlite:///mlflow.db` |
| ≥ 3 corridas con un modelo base simple, parámetros, métricas y modelo como artefacto | ✅ 16 corridas (base: tasa por plan × segmento) |
| Tabla `mlflow.search_runs()` en el notebook + corrida elegida y por qué | ✅ `solucion.ipynb` §4, ejecutado en Colab |
| No subir `mlruns` | ✅ en `.gitignore` (también `mlflow.db`) |

## Entregables del repositorio

| Archivo | Estado | Nota |
|---|---|---|
| `README.md` con botón "Open in Colab" | ✅ | Botón verificado; apunta a `adrianfelipe812/prueba-conecta` |
| `solucion.ipynb` | ✅ | Salidas de la corrida en Colab real (21/21 celdas, 0 errores, pandas 2.2.3); botón Open in Colab al inicio |
| `requirements.txt` (solo lo que Colab no trae) | ✅ | `mlflow` |
| `predicciones.csv` | ✅ | Generado en Colab: 1.558 filas, 522 a contactar, validado |
| `MEMO.md` (1 página, no técnica) | ✅ | |
| `BITACORA_IA.md` (uso de IA + horas) | ✅ | 3,9 h |
| Repo público en GitHub | ✅ | https://github.com/adrianfelipe812/prueba-conecta · auditado: sin datos ni textos de clientes, sin rutas personales; un solo commit (repositorio recreado para eliminar versiones anteriores) |
| `storytelling.html` (adicional) | ✅ | Historia de negocio interactiva en 3 actos con simulador |

## Requisitos técnicos

| Requisito | Estado | Nota |
|---|---|---|
| No subir los datos (la prueba es confidencial) | ✅ | `.gitignore` excluye `raw/`, `lake/`, `*.db`, `*.zip`, `*.jsonl*`; sin textos de tickets ni filas en salidas (escáner: 0 coincidencias); etiquetas NLP por huella |
| Módulos `.py` permitidos, nada que correr fuera de Colab | ✅ | La primera celda clona el repo público e instala `requirements.txt` |
| Correr en Colab sin errores | ✅ | Corrida completa en Colab real; mismas decisiones que en local |
| Todo gratuito, sin ejecutables, sin API keys | ✅ | |
| Bonus (Streamlit / agente / mini app) | ➖ | No incluido. `storytelling.html` trae un simulador de la campaña, pero no reemplaza el bonus |

## Decisiones y revisiones del autor

- ✅ Mensualidad actual = precio vigente del plan.
- ✅ Etiquetas NLP de referencia revisadas y confirmadas.
- ✅ Cortes adicionales evaluados: no se usan (anotación).
- ✅ Criterio de selección = costo de la decisión; modelo = ensamble logística L1 + Random Forest 50/50.
- ✅ Regularización: Ridge, Lasso y Elastic Net empatan; se mantiene Lasso (`experimento_elasticnet.py`).
- ✅ No recontactar a quien ya recibió una oferta (costo medido y documentado).
- ✅ Hallazgo abierto: Android 5.4.0.
- ✅ Frecuencia: reentrenar trimestral (o por alarma), puntuar mensual.
- **Plazo:** la entrega se retrasó respecto a los 2 días del enunciado por un motivo de fuerza mayor; se comunica al enviar el enlace.

## Posibles mejoras (fuera del alcance entregado)

- Modelo de alerta temprana sin variables tardías (contacto previo, ticket de cancelación), para llamar antes de que el cliente pida irse.
- Ajuste de hiperparámetros con validación anidada (GroupKFold dentro, origen móvil fuera).
- Grupo de control en la campaña para medir el efecto real de la oferta.

## Comandos útiles

```bash
cd prueba-conecta
.venv/Scripts/python.exe -m conecta.pipeline                # bronze → silver → gold → calidad
.venv/Scripts/python.exe -m conecta.storytelling            # regenera storytelling.html
.venv/Scripts/python.exe -m conecta.experimento_frecuencia  # simulación de frecuencia de actualización
.venv/Scripts/python.exe -m conecta.experimento_elasticnet   # Ridge vs Lasso vs Elastic Net
.venv/Scripts/mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```
