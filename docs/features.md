# Diccionario de variables del modelo (gold/features)

Todas se calculan **al corte** (point-in-time) con `conecta.gold.features_corte`, con la misma función para entrenar (cortes 2025-03-31, 06-30 y 09-30) y para puntuar (2025-12-31).

| Grupo | Variable | Definición |
|---|---|---|
| Perfil | `antiguedad_dias` | Días desde el alta |
| | `edad`, `flag_edad_invalida` | Edad (NaN si era centinela 0/99/999) y bandera |
| | `segmento`, `ciudad`, `canal_adquisicion`, `plataforma_principal` | Categóricas del CRM (ciudad normalizada) |
| Contrato | `plan`, `precio_plan` | Plan vigente al corte y su precio mensual vigente |
| | `dias_en_plan_actual`, `n_cambios_plan`, `bajo_de_plan_antes` | Permanencia en el plan e historial de cambios |
| | `afectado_aumento_basico` | Cliente Básico que pagó 19.900 y vivió la subida a 23.900 (desde abr-2025) |
| | `contacto_retencion_previo` | Contacto de retención con fecha ≤ corte (excluye las 76 fechas anteriores al alta) |
| Facturación | `facturas_impagas`, `facturas_impagas_3m` | Facturas vencidas sin pago **conocido al corte** |
| | `atraso_prom_6m`, `pagos_tarde_6m`, `dias_desde_ultimo_pago` | Comportamiento de pago (solo pagos con fecha ≤ corte) |
| Tickets + NLP | `tickets_30d`, `tickets_90d`, `tickets_total`, `dias_desde_ultimo_ticket` | Volumen y recencia |
| | `tickets_cancelar_30d`, `tickets_amenaza_90d`, `tickets_bajar_plan_90d` | Intención detectada por NLP |
| | `tickets_sarcasmo_90d`, `frustracion_max_90d`, `tickets_recurrente_90d`, `tickets_precio_90d` | Señales NLP |
| | `tickets_app_90d`, `tickets_cobro_90d`, `tickets_abiertos` | Tema y tickets abiertos al corte |
| | `csat_min_90d`, `csat_prom_90d` | Satisfacción de tickets resueltos antes del corte (NaN si no hay) |
| Uso de la app | `logins_30d`, `logins_31_90d`, `tendencia_logins` | Actividad reciente y su tendencia |
| | `uso_30d`, `duracion_uso_30d` | Uso de funcionalidades |
| | `crash_30d`, `errores_30d`, `usa_version_540` | Errores de la app y uso de la versión 5.4.0 |
| | `dias_desde_ultimo_evento`, `sin_eventos_90d` | Recencia de uso |

**Excluidas a propósito (fuga de información):** `estado_actual`, `motivo_cancelacion`, `fecha_fin`, contactos con fecha posterior al corte y pagos registrados después del corte.
