"""Constantes y supuestos del proyecto. Cambiar aquí (cambio en vivo de la sustentación)."""
from pathlib import Path
import sys

# --- Fechas -------------------------------------------------------------------
FECHA_HOY = "2025-12-31"                     # "hoy" según el enunciado
CORTES_ENTRENAMIENTO = ["2025-03-31", "2025-06-30", "2025-09-30"]
HORIZONTE_DIAS = 90
ZONA_HORARIA = "America/Bogota"              # todas las marcas de tiempo se llevan a hora local

# --- Negocio (enunciado) --------------------------------------------------------
PRECIO_PLAN = {"Basico": 23_900, "Estandar": 34_900, "Premium": 49_900, "Familiar": 59_900}
COSTO_CONTACTO = 25_000
TASA_EXITO_OFERTA = 0.40
MESES_VALOR_RETENCION = 12
# Decisión del negocio (2026-11-22): no volver a contactar a quien ya recibió una oferta de
# retención (contacto con fecha <= corte). Costo medido en validación: −2,1 M (jun) y −2,7 M (sep)
# bajo los supuestos del enunciado. Poner en False para revertir.
EXCLUIR_CONTACTADOS_PREVIOS = True

# --- Limpieza -----------------------------------------------------------------
EDAD_VALIDA = (18, 90)                       # 0, 99 y 999 son centinelas -> NaN
CIUDADES = {                                 # variante normalizada (minúsculas, sin tildes) -> canónica
    "bogota": "Bogotá", "bogota d.c.": "Bogotá",
    "medellin": "Medellín",
    "cali": "Cali", "santiago de cali": "Cali",
    "barranquilla": "Barranquilla", "b/quilla": "Barranquilla",
    "bucaramanga": "Bucaramanga",
    "pereira": "Pereira",
}
# Categoría del esquema antiguo de tickets (A) -> categoría principal del esquema nuevo (B)
CATEGORIA_PRINCIPAL = {
    "lentitud": "tecnico", "app_se_cierra": "tecnico",
    "cambio_clave": "cuenta",
    "cobro_doble": "facturacion", "no_reconoce_cobro": "facturacion",
    "cancelar": "comercial", "cambio_plan": "comercial",
}

# --- Rutas --------------------------------------------------------------------
EN_COLAB = "google.colab" in sys.modules
RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/content/datos") if EN_COLAB else RAIZ / "raw" / "datos_conecta_extraido"
LAKE_DIR = Path("/content/lake") if EN_COLAB else RAIZ / "lake"
