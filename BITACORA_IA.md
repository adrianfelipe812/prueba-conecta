# Bitácora de uso de IA

Herramienta principal: Claude (Claude Code).

## Tiempo invertido

| Fecha | Sesión | Horas | Actividad |
|---|---|---|---|
| 2026-11-19 | ~20:30 – 20:50 | 0,3 | Revisión de seguridad del zip, extracción verificada, lectura del enunciado, entorno Python 3.12 + MLflow |
| 2026-11-19 | ~20:50 – 21:05 | 0,3 | Perfilado de calidad de las 7 fuentes, arquitectura medallion (bronze/silver/gold), reporte de calidad, notebook 01 |
| 2026-11-22 | ~10:15 – 10:45 | 0,5 | NLP de tickets con sarcasmo en silver y validación (referencia, desarrollo, prueba, pysentimiento, churn); experimento de cortes adicionales |
| 2026-11-22 | ~10:45 – 11:25 | 0,7 | Features gold, comparación de 15 modelos por costo de decisión, ensamble elegido, MLflow, `predicciones.csv`, exclusión de contactados, revisión de pendientes vs enunciado |
| 2026-11-22 | ~11:25 – 11:45 | 0,3 | `solucion.ipynb` (partes 1–7 + MLflow), hallazgo Android 5.4.0, producción, prueba con versiones tipo Colab |
| 2026-11-22 | ~11:45 – 11:55 | 0,2 | `MEMO.md` y `README.md` |
| 2026-11-22 | ~11:55 – 12:15 | 0,3 | Repo público en GitHub (auditoría de datos) y ejecución completa en Colab |
| 2026-11-22 | ~12:15 – 12:25 | 0,2 | Descarga de las salidas de Colab y de `predicciones.csv`, validación y subida |
| 2026-11-22 | ~12:25 – 12:45 | 0,3 | Storytelling de negocio en HTML (criterios del curso de visualización) y guía privada de sustentación |
| 2026-11-22 | ~12:45 – 13:00 | 0,3 | Análisis de frecuencia de actualización del modelo, estado final y cierre del repositorio |
| 2026-11-22 | ~13:00 – 13:20 | 0,3 | Revisión de confidencialidad: retirar textos de tickets y filas de datos, y limpiar el historial público |
| 2026-11-22 | ~13:20 – 13:30 | 0,2 | Ajustes de la guía de sustentación y prueba de Elastic Net frente a Ridge y Lasso |

**Total aproximado: 3,9 horas.**

## Momentos importantes

### 1. Revisión de seguridad y extracción de los archivos

- **Qué pedí:** una herramienta para revisar los archivos recibidos y extraerlos de forma segura.
- **Qué acepté:** escaneo sin extraer (rutas, zip bombs, contenido activo en PDF, comandos en el notebook) y extracción a una carpeta de solo lectura con hashes.
- **Qué corregí / rechacé:** el primer escaneo marcó el PDF como "contenido activo" (`/AA`); al revisarlo era un falso positivo (nombre de fuente `/AAAAAA+...`) y se ajustó la detección.
- **Por qué:** quiero trazabilidad y no ejecutar nada de origen externo sin leerlo antes.

### 2. Entorno

- **Qué pedí:** preparar el entorno local.
- **Qué decidí:** Python 3.12 en lugar del 3.14 del sistema, aunque las librerías tenían soporte para 3.14, para parecerse más a Colab.
- **Qué se corrigió:** MLflow 3.18 rechazó el almacenamiento en carpeta `./mlruns`; se cambió a SQLite local.

### 3. Calidad de datos y arquitectura medallion

- **Qué pedí:** explorar la calidad de los datos y organizarlos en capas bronze/silver/gold para revisar cada fase.
- **Qué acepté:** un catálogo de problemas con evidencia (hoy 15, P01–P15), la limpieza en silver con banderas y un reporte automático que detiene el pipeline si algo no queda resuelto.
- **Qué verifiqué:** que la definición de churn propuesta (fin de la última suscripción en 90 días) reproduce exactamente las 4.810 etiquetas dadas, y que los gráficos no escondieran información (el de tickets `app_se_cierra` se corrigió para mostrar todo 2025).

### 4. Mensualidad actual

- **Qué decidí:** usar los precios mensuales vigentes del enunciado junto con sus supuestos de negocio, no el monto histórico de las facturas.

### 5. NLP de tickets y sarcasmo

- **Qué pedí:** un NLP preciso, validar el sarcasmo encontrado e integrarlo en la capa silver.
- **Qué acepté:** reglas transparentes (sarcasmo = elogio o risa + evento negativo, sin señal de "resuelto") con columnas `nlp_*` en silver, y controles de concordancia con un set de referencia etiquetado a mano.
- **Qué detecté y corregí:** la primera versión acertaba el 100 % en los 38 textos reales pero solo el 80 % en frases nuevas, porque se había sobreajustado a las plantillas. Se ampliaron los léxicos con reglas generales y se creó un set de **prueba** nuevo para medir sin trampa. En ese set la detección de sarcasmo queda en 79 %: es el límite real del método y lo documento.
- **Qué rechacé:** usar pysentimiento (modelo preentrenado) en el pipeline. Rinde peor que las reglas en los tres sets, lee como positivas las quejas sarcásticas y pesa ~1 GB. Lo dejo solo como validación externa.
- **Qué revisé yo:** las 38 etiquetas de referencia que propuso la IA (base de la validación); las confirmo.
- **Lo que aprendí para el modelo:** la intención (cancelar, amenaza, bajar de plan) predice más que el sentimiento; el sentimiento solo engaña.

### 6. Cortes de entrenamiento adicionales

- **Qué pedí:** validar si generar cortes adicionales mejora el modelo; si no es relevante, dejarlo como anotación.
- **Resultado:** no es relevante (mensuales 2025: +0,004 a +0,010 AUC, IC 95 % incluye 0; desde 2023–2024: empeora). Queda como anotación y se usan los 3 cortes oficiales.

### 7. Modelos y elección por costo de la decisión

- **Qué pedí:** probar todos los modelos y elegir el mejor teniendo en cuenta los "pesos", es decir, el costo de cada decisión según los supuestos de negocio (contacto 25.000, éxito 40 %, valor de 12 meses del precio vigente).
- **Qué acepté:** criterio = valor neto real de la regla "contactar si p > 25.000 / (4,8 × precio)" en validación temporal (dos pliegues), con probabilidades calibradas; 15 candidatos registrados en MLflow.
- **Qué evalué y descarté:** el ajuste por cambio de prevalencia (no mejora de forma consistente) y el entrenamiento ponderado por valor (rinde menos).
- **Decisión final:** Random Forest y logística L1 empataban (bootstrap). Elegí su promedio 50/50 con pesos fijados de antemano, para no sobreajustar a la validación; supera a ambos.
- **Qué se corrigió en el camino:** la línea base estaba mal calibrada; la red neuronal estaba mal configurada (AUC 0,46); los contactos con fecha anterior al alta contaban como contacto; la logística L1 no tenía semilla (resultados no reproducibles).

### 8. Exclusión de clientes ya contactados

- **Qué decidí:** no volver a contactar a quien ya recibió una oferta de retención.
- **Qué me mostró la IA antes de implementarlo:** bajo los supuestos del enunciado, la exclusión cuesta valor (−2,1 M en junio y −2,7 M en septiembre), porque esos clientes tienen ~55 % de churn y la oferta se asume efectiva al 40 % para todos.
- **Por qué la mantengo:** el 94 % de los contactados históricos canceló de todas formas; repetir la oferta es poco creíble para el negocio. Quedó como opción reversible en `config.py`.

### 9. Notebook principal, hallazgo y producción

- **Qué pedí:** armar `solucion.ipynb` con lo necesario.
- **Qué propuso la IA:** un notebook autocontenido para Colab (clona el repo, descomprime el zip validando rutas, corre el pipeline, los modelos, MLflow y las predicciones). Como hallazgo abierto propuso la app Android 5.4.0, con evidencia de logs, tickets y CRM, y escribió la sección de producción.
- **Qué verifiqué:** que corre de principio a fin sin errores, también con versiones tipo Colab (pandas 2.2.2, scikit-learn 1.6.1) y luego en Colab real (sección 11).

### 10. MEMO y README

- **Qué pedí:** el memo para la gerencia y el README.
- **Qué propuso la IA:** un memo de una página sin jerga (qué hacer, impacto, hallazgo de la app, limitaciones) y un README con el botón de Colab, instrucciones y la estructura del repo.
- **Resultado:** el botón y la celda de clonado apuntan al repositorio público.

### 11. Repositorio y Colab

- **Qué pedí:** preparar el repo manteniendo los datos anonimizados, subirlo a mi GitHub y correr el notebook en Colab.
- **Qué se verificó antes de subir:** que no hubiera archivos de datos, secretos ni correos; se quitaron de las salidas las rutas locales con mi usuario y los IDs de cliente de un ejemplo. Los commits usan mi correo *noreply* de GitHub.
- **Resultado en Colab:** corre de principio a fin sin errores (pandas 2.2.3); mismas decisiones de contacto (522) y misma elección del modelo.
- **Entrega:** el `solucion.ipynb` y el `predicciones.csv` del repo son los de la corrida en Colab (validados: sin errores, sin datos de mi cuenta en los metadatos, 1.558 filas). Botón Open in Colab en el README y al inicio del notebook.
- **Qué corrigió la IA al revisar:** una línea en blanco dentro de la tabla de tiempos la partía en GitHub; se quitó y se revisó el formato de todos los `.md`.

### 12. Storytelling para el negocio
- **Qué pedí:** un HTML que cuente los hallazgos al negocio con el rigor de mis clases de visualización de datos.
- **Qué acepté:** estructura en 3 actos con una Gran Idea; Bing–Bang–Bongo (título que afirma, gráfico que guía, conclusión visible); foco vs. contexto con la paleta del curso (azul, coral y gris, validada para daltonismo); gráfico según la intención (líneas para tendencia, barras desde cero y ordenadas para comparar, barras apiladas para parte-todo); detalle bajo demanda e intervalos de confianza; simulador de supuestos sin identificadores de clientes.
- **Qué corregí al revisarlo:** (1) el eje "fecha de corte" hacía parecer que la subida de Android ocurría antes del lanzamiento de la 5.4.0; se cambió a la ventana que se mide; (2) un titular decía "casi tres veces más" y no era comparable; ahora se calcula solo, con la misma política en ambos lados (4 veces); (3) formato numérico en español.

### 13. Frecuencia de actualización del modelo
- **Qué pregunté:** si el modelo debía actualizarse periódicamente, preocupado porque la llamada de retención llega solo 3–19 días antes de la baja.
- **Qué hizo la IA:** en vez de opinar, simuló volver a puntuar cada mes con el mismo modelo (`conecta/experimento_frecuencia.py`).
- **Qué encontramos:** repuntuar el 31-oct y el 30-nov suma 331 llamadas y casi nada de valor (+0,8 M y −1,0 M). El 41 % de quienes el modelo no marcó solo dio señal después, pidiendo cancelar, y esos clientes se van en 1–14 días.
- **Qué decidí:** reentrenar cada trimestre (o antes si salta una alarma) y puntuar cada mes. La mejora de fondo es un modelo de alerta temprana con señales anticipadas; queda como siguiente paso.
- **Límite que acepto:** las rondas tardías se observan menos de 90 días, así que su valor está algo subestimado.

### 14. Confidencialidad del repositorio
- **Qué motivó la revisión:** el correo de la prueba pide no compartir la prueba ni los datos, y el repositorio es público.
- **Qué hizo la IA:** un escáner que busca, en todos los archivos versionados, fragmentos de los textos reales de tickets e IDs. Encontró contenido en 11 de 35 archivos; el caso principal era el notebook 02, que imprimía los 38 textos.
- **Qué se cambió:** paráfrasis propias en lugar de citas (MEMO, README, documentos, storytelling y comentarios de código); notebooks auxiliares que muestran huellas y conteos en lugar de textos y filas. En `solucion.ipynb`, la celda de ejemplos de sarcasmo pasó a un resumen por categoría. Su salida se generó localmente con el nuevo código y es idéntica en lo verificable a la de Colab (mismo total: 974 tickets sarcásticos), porque el NLP es determinista. El resto de las salidas siguen siendo las de Colab.
- **Resultado:** el escáner da 0 coincidencias en los 35 archivos.
- **Historial:** las versiones anteriores seguían visibles en el historial público. Decidí dejar un solo commit limpio y recrear el repositorio, porque GitHub mantiene accesibles por enlace directo los commits que ya no pertenecen a ninguna rama. El historial antiguo quedó respaldado solo en mi equipo.

### 15. Ridge, Lasso y Elastic Net
- **Qué pregunté:** por qué no se usaron Ridge, Lasso y Elastic Net.
- **Qué aclaró la IA:** Ridge (L2) y Lasso (L1) sí estaban, como regresión logística penalizada; faltaba Elastic Net.
- **Qué se probó:** las tres con el mismo protocolo (dos pliegues temporales, calibración, valor neto) en `conecta/experimento_elasticnet.py`. Elastic Net con 25 % de L1 llegó a 35,2 % del máximo, frente a 34,2 % de Lasso y 34,1 % de Ridge; el ensamble elegido sigue arriba (36,5 %).
- **Qué decidí:** mantener Lasso. La ventaja de Elastic Net es pequeña, está inflada porque elegí la mejor de tres mezclas mirando la validación, y agrega un hiperparámetro que no conviene afinar con dos pliegues.

