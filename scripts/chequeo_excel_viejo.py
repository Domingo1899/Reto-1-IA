"""
chequeo_excel_viejo.py — Radiografía de los archivos, SIN cargar los datos.

Antes de analizar nada hay que saber qué tenemos:
  - ¿Cuántas hojas tiene cada Excel? (pandas lee solo la primera por defecto)
  - ¿Cómo se llaman exactamente las columnas?
  - ¿Las columnas de 2025 son las mismas que las de 2026?

No leemos los datos completos (son decenas de MB): solo la estructura.
"""

import re
from pathlib import Path

import pandas as pd

CARPETA = Path("data/raw/entrega-01-original")


def identificar(nombre: str) -> tuple[str | None, int | None]:
    """
    Deduce (poblacion, anio) del nombre del archivo, sea cual sea.

    'docente2025viejo'      -> ('docentes', 2025)
    'estudiantes_2026.xlsx' -> ('estudiantes', 2026)
    'ESTUD 2026 final'      -> ('estudiantes', 2026)

    Así el script no depende de que respetemos una convención de nombres.
    """
    n = nombre.lower()

    if "docent" in n:
        poblacion = "docentes"
    elif "estudiant" in n or "alumn" in n:
        poblacion = "estudiantes"
    else:
        poblacion = None

    # \d{4} = cuatro dígitos seguidos. Buscamos 2025 o 2026.
    match = re.search(r"20\d{2}", n)
    anio = int(match.group()) if match else None

    return poblacion, anio


def inspeccionar(ruta: Path) -> dict:
    """Devuelve {hoja: [columnas]} sin cargar el archivo entero."""
    poblacion, anio = identificar(ruta.stem)

    print("=" * 70)
    print(f"ARCHIVO: {ruta.name}   ({ruta.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> interpretado como: {poblacion} / {anio}")
    print("=" * 70)

    if poblacion is None or anio is None:
        print("  !! No pude deducir población y/o año del nombre.")

    # ExcelFile abre el índice del archivo, NO las filas: es instantáneo.
    xl = pd.ExcelFile(ruta)
    print(f"Hojas: {xl.sheet_names}")

    estructura = {}
    for hoja in xl.sheet_names:
        muestra = pd.read_excel(ruta, sheet_name=hoja, nrows=5)  # solo 5 filas
        estructura[hoja] = list(muestra.columns)
        print(f"\n  Hoja '{hoja}': {len(muestra.columns)} columnas")
        for i, col in enumerate(muestra.columns, 1):
            print(f"    {i:2d}. {col}")
    print()

    return {"poblacion": poblacion, "anio": anio, "hojas": estructura}


def main():
    archivos = sorted(CARPETA.glob("*.xlsx"))
    if not archivos:
        print(f"No encontré archivos .xlsx en {CARPETA}")
        return

    info = [inspeccionar(r) for r in archivos]

    # ---------- Comparación de esquemas entre años ----------
    print("=" * 70)
    print("¿SON COMPARABLES 2025 Y 2026?")
    print("=" * 70)

    for poblacion in ["docentes", "estudiantes"]:
        d25 = next((x for x in info if x["poblacion"] == poblacion and x["anio"] == 2025), None)
        d26 = next((x for x in info if x["poblacion"] == poblacion and x["anio"] == 2026), None)

        print(f"\n{poblacion.upper()}")
        if d25 is None or d26 is None:
            # Un chequeo que no puede correr AVISA. Nunca se calla.
            print("  !! Falta el archivo de 2025 o de 2026: no puedo comparar.")
            continue

        cols25 = set(d25["hojas"].get("Datos", []))
        cols26 = set(d26["hojas"].get("Datos", []))

        solo25 = sorted(cols25 - cols26)
        solo26 = sorted(cols26 - cols25)

        if not solo25 and not solo26:
            print("  OK: mismas columnas en ambos años.")
        else:
            print(f"  Solo en 2025: {solo25}")
            print(f"  Solo en 2026: {solo26}")
            print("  >>> HALLAZGO: el esquema cambió entre años.")
            print("      Si concatenamos sin mapear, pandas crea columnas separadas")
            print("      y los datos de un año quedan vacíos en la columna del otro.")


if __name__ == "__main__":
    main()