"""
chequeo_nulos.py — Radiografía de los valores faltantes.

CUIDADO CON ESTE CHEQUEO
Un nulo no es siempre un error. En este dataset sabemos (por el diccionario, D2):

  - CONTEXTO  aplica SOLO a primaria     -> nulo en secundaria = ESPERADO
  - IVSMEDIA  aplica SOLO a UTU/Secundaria -> nulo en primaria = ESPERADO

Un chequeo que grita "55% de nulos en CONTEXTO!" sin entender eso es un chequeo
que nadie va a mirar. Peor: si alguien "limpia" esos nulos borrando filas, se
lleva puesta media base.

Por eso este script hace dos cosas:
  1. Cuenta los nulos por columna (lo básico)
  2. CRUZA los nulos de contexto/ivsmedia con el subsistema (Rubro), para
     separar los esperados de los que no lo son.

También detecta "nulos disfrazados": textos como '-', 's/d', 'Sin clasificar'
que pandas lee como valores normales pero que en realidad significan "no hay dato".
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")

# Textos que significan "vacío" aunque pandas los lea como texto común
NULOS_DISFRAZADOS = {
    "", "-", "--", "s/d", "sd", "n/a", "na", "nan", "null", "none",
    "sin dato", "sin datos", "desconocido", ".", "?", "no aplica",
}


def norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def col(df, nombre):
    """Encuentra la columna sin importar mayúsculas: 'Zona' o 'ZONA'."""
    return next((c for c in df.columns if norm(c) == norm(nombre)), None)


def resumen_nulos(df: pd.DataFrame, tabla: str):
    total = len(df)
    print(f"\n{'=' * 78}")
    print(f"{tabla}   ({total:,} filas)")
    print("=" * 78)

    filas = []
    for c in df.columns:
        nulos_reales = int(df[c].isna().sum())
        disfrazados = int(df[c].map(lambda v: norm(v) in NULOS_DISFRAZADOS and not pd.isna(v)).sum())
        faltantes = nulos_reales + disfrazados

        ejemplos = ""
        if disfrazados:
            vals = df[c][df[c].map(lambda v: norm(v) in NULOS_DISFRAZADOS and not pd.isna(v))]
            ejemplos = " | ".join(repr(x) for x in vals.unique()[:3])

        filas.append({
            "columna": c,
            "nulos": nulos_reales,
            "disfrazados": disfrazados,
            "total_faltante": faltantes,
            "%": round(100 * faltantes / total, 1),
            "ejemplos_disfrazados": ejemplos,
        })

    res = pd.DataFrame(filas).sort_values("%", ascending=False)
    # solo mostramos las columnas que tienen algo
    con_nulos = res[res["total_faltante"] > 0]

    if con_nulos.empty:
        print("  Sin valores faltantes en ninguna columna.")
    else:
        print(con_nulos.to_string(index=False))

    print(f"\n  Columnas SIN ningún faltante: "
          f"{list(res[res['total_faltante'] == 0]['columna'])}")

    return res


def nulos_por_subsistema(df: pd.DataFrame, tabla: str):
    """
    LA PARTE IMPORTANTE.
    Cruza los nulos de contexto/ivsmedia con el subsistema, para separar
    los faltantes ESPERADOS (por diseño) de los que son un problema real.
    """
    c_rubro = col(df, "Rubro")
    if c_rubro is None:
        return

    print(f"\n--- ¿Los nulos de contexto/ivsmedia son los esperados? ---")
    print("    DGEIP=Primaria  DGES=Secundaria  DGETP=UTU")
    print("    Esperado: CONTEXTO solo en DGEIP | IVSMEDIA solo en DGES y DGETP\n")

    for nombre in ["Contexto Sociocultural", "IvsMedia"]:
        c = col(df, nombre)
        if c is None:
            continue

        tmp = df[[c_rubro, c]].copy()
        tmp["rubro"] = tmp[c_rubro].map(norm)
        tmp["tiene_dato"] = tmp[c].notna() & (tmp[c].map(norm) != "")

        g = tmp.groupby("rubro")["tiene_dato"].agg(
            filas="size", con_dato="sum"
        ).reset_index()
        g["% con dato"] = (100 * g["con_dato"] / g["filas"]).round(1)

        print(f"  {nombre}:")
        for _, r in g.iterrows():
            # marcamos lo que NO cuadra con lo esperado
            esperado_primaria = nombre.lower().startswith("contexto")
            deberia_tener = (r["rubro"] == "dgeip") if esperado_primaria else (r["rubro"] in ("dges", "dgetp"))

            if deberia_tener and r["% con dato"] < 90:
                marca = "  <<< PROBLEMA: debería tener dato y no lo tiene"
            elif not deberia_tener and r["% con dato"] > 5:
                marca = "  <<< RARO: no debería tener dato y lo tiene"
            else:
                marca = "  (ok)"

            print(f"    {r['rubro']:<7} {int(r['filas']):>8,} filas | "
                  f"{r['% con dato']:>5.1f}% con dato{marca}")
        print()


def main():
    pd.set_option("display.width", 200)

    for ruta in sorted(CARPETA.glob("*.parquet")):
        df = pd.read_parquet(ruta)
        resumen_nulos(df, ruta.stem)
        nulos_por_subsistema(df, ruta.stem)


if __name__ == "__main__":
    main()