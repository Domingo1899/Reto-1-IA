"""
convertir_a_parquet.py — Convierte los Excel a Parquet una sola vez.

POR QUÉ
Leer los 4 Excel tarda ~2 minutos. Vamos a iterar sobre estos datos decenas de
veces (auditar, limpiar, analizar). 2 min x 30 iteraciones = 1 hora perdida.
Parquet carga en segundos.

Los Excel originales NO se tocan: quedan en data/raw/ como fuente de verdad.
El Parquet es una copia intermedia, y por eso va a data/interim/.

IMPORTANTE: leemos todo como TEXTO (dtype=str). Suena raro, pero es deliberado:
si dejamos que pandas adivine los tipos, "arregla" cosas en silencio (convierte
un día de acceso vacío en NaN, un ID '007' en 7). Queremos ver el dato TAL CUAL
vino, y recién después decidir qué hacer. Auditar primero, convertir después.
"""

import re
import time
from pathlib import Path

import pandas as pd

ORIGEN = Path("data/raw/entrega-02-corregida")
DESTINO = Path("data/interim/entrega-02-corregida")


def identificar(nombre: str):
    n = nombre.lower()
    poblacion = "docentes" if "doc" in n else ("estudiantes" if "estu" in n else None)
    m = re.search(r"20\d{2}", n)
    return poblacion, (int(m.group()) if m else None)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)

    archivos = sorted(ORIGEN.glob("*.xlsx"))
    if not archivos:
        print(f"No encontré .xlsx en {ORIGEN}")
        return

    for ruta in archivos:
        poblacion, anio = identificar(ruta.stem)
        if not poblacion or not anio:
            print(f"!! No pude identificar {ruta.name}, lo salteo.")
            continue

        t0 = time.time()
        df = pd.read_excel(ruta, sheet_name="Sheet 1", dtype=str)

        # nombre estandarizado: acá SÍ imponemos la convención, en la copia,
        # nunca en el original
        salida = DESTINO / f"{poblacion}_{anio}.parquet"
        df.to_parquet(salida, index=False)

        mb_in = ruta.stat().st_size / 1e6
        mb_out = salida.stat().st_size / 1e6
        print(
            f"{ruta.name:<28} -> {salida.name:<24} "
            f"{len(df):>8,} filas | {mb_in:>5.1f} MB -> {mb_out:>4.1f} MB "
            f"| {time.time() - t0:.1f}s"
        )

    print(f"\nListo. A partir de ahora los scripts leen de {DESTINO}/")


if __name__ == "__main__":
    main()