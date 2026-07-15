"""
metricas_acceso.py — La primera respuesta al reto.

LA PREGUNTA
Ceibal dice: "en 2026 accedieron menos usuarios que en 2025". Nosotros
tenemos que averiguar si eso es un cambio de COMPORTAMIENTO o un cambio
de POBLACIÓN. Son cosas muy distintas:

  - Menos usuarios registrados  -> hay menos gente. El conteo baja solo.
  - Menor tasa de acceso        -> la gente que está, entra menos. ESO es
                                   un problema de la plataforma.

Por eso calculamos TRES cosas, no una:

  1. N REGISTRADOS  : cuánta gente hay en la base (el denominador)
  2. TASA DE ACCESO : % de esa gente que entró al menos 1 día
  3. INTENSIDAD     : cuántos días entró, entre los que SÍ entraron

Ejemplo de por qué hacen falta las tres:
  Si la tasa se mantiene pero hay 8% menos gente -> no es problema de CREA,
  es matrícula. Si la tasa cae -> ahí sí, algo pasó con la plataforma.
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")

MESES = {"Dias4": "abril", "Dias5": "mayo", "Dias6": "junio"}

# DECISIÓN (documentar en decisiones.md):
# "usuario que accedió" = al menos 1 día de acceso en el trimestre.
# Es la lectura más directa del PDF ("cantidad de usuarios que accedieron").
# Más abajo probamos también otros umbrales, para ver si la conclusión cambia.
UMBRAL = 1


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def preparar(df: pd.DataFrame, poblacion: str, anio: int) -> pd.DataFrame:
    """Normaliza y calcula las columnas derivadas que necesitamos."""
    df = df.copy()

    # 1. Nombres de columna a minúsculas -> resuelve Zona/ZONA, IvsMedia/IVSMEDIA
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]

    # 2. Valores de texto a minúsculas -> resuelve 'Urbana' vs 'URBANA'
    for col in ["zona", "tipo_centro", "rubro", "dept_nombre", "ciclo", "sexo"]:
        if col in df.columns:
            df[col] = df[col].map(norm)

    # 3. QUITAR DUPLICADOS EXACTOS
    #    Encontramos 6.449 en docentes 2025 y solo 641 en 2026. Si no los
    #    sacamos, 2025 queda inflado y la caída hacia 2026 se exagera.
    antes = len(df)
    df = df.drop_duplicates()
    dups = antes - len(df)

    # 4. Días de acceso a número
    for col in MESES:
        df[norm(col)] = pd.to_numeric(df[norm(col)], errors="coerce")

    cols_dias = [norm(c) for c in MESES]
    df["dias_total"] = df[cols_dias].sum(axis=1, min_count=1)
    df["accedio"] = df["dias_total"] >= UMBRAL

    df["anio"] = anio
    df["poblacion"] = poblacion

    print(f"  {poblacion} {anio}: {antes:,} filas -> {len(df):,} "
          f"(se quitaron {dups:,} duplicados exactos)")
    return df


def tasa(df, por):
    """% de usuarios que accedieron, agrupado por lo que le pidas."""
    g = df.groupby(por, dropna=False).agg(
        n_registrados=("accedio", "size"),
        n_accedieron=("accedio", "sum"),
    ).reset_index()
    g["tasa_%"] = (100 * g["n_accedieron"] / g["n_registrados"]).round(1)
    return g


def comparar(df, dimension=None):
    """Compara 2025 vs 2026 en la dimensión que le pases."""
    por = ["poblacion", "anio"] + ([dimension] if dimension else [])
    t = tasa(df, por)

    idx = ["poblacion"] + ([dimension] if dimension else [])
    piv = t.pivot_table(index=idx, columns="anio",
                        values=["n_registrados", "tasa_%"]).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]

    piv["var_registrados_%"] = (
        100 * (piv["n_registrados_2026"] - piv["n_registrados_2025"])
        / piv["n_registrados_2025"]
    ).round(1)
    piv["var_tasa_pp"] = (piv["tasa_%_2026"] - piv["tasa_%_2025"]).round(1)

    return piv[[*idx, "n_registrados_2025", "n_registrados_2026", "var_registrados_%",
                "tasa_%_2025", "tasa_%_2026", "var_tasa_pp"]]


def main():
    pd.set_option("display.width", 200)

    print("Cargando y preparando...")
    partes = []
    for ruta in sorted(CARPETA.glob("*.parquet")):
        poblacion, anio = ruta.stem.split("_")
        partes.append(preparar(pd.read_parquet(ruta), poblacion, int(anio)))
    df = pd.concat(partes, ignore_index=True)

    # ============================================================
    print("\n" + "=" * 78)
    print("1. LA PREGUNTA CENTRAL: ¿menos gente, o menos uso?")
    print("=" * 78)
    print(comparar(df).to_string(index=False))
    print("""
  CÓMO LEERLO:
    var_registrados_% -> cambió la cantidad de gente en la base (denominador)
    var_tasa_pp       -> cambió el % que entró a CREA (comportamiento)

    Si var_tasa_pp ~ 0  -> la caída es de POBLACIÓN, no de uso.
    Si var_tasa_pp < 0  -> la gente que está, entra menos. Eso sí es CREA.
    """)

    # ============================================================
    print("\n" + "=" * 78)
    print("2. POR SUBSISTEMA (Rubro)   DGEIP=Primaria DGES=Secundaria DGETP=UTU")
    print("=" * 78)
    print(comparar(df, "rubro").to_string(index=False))

    # ============================================================
    print("\n" + "=" * 78)
    print("3. EVOLUCIÓN MES A MES (¿la caída se concentra en algún mes?)")
    print("=" * 78)
    filas = []
    for col, mes in MESES.items():
        c = norm(col)
        tmp = df.copy()
        tmp["accedio_mes"] = tmp[c] >= UMBRAL
        g = tmp.groupby(["poblacion", "anio"]).agg(
            n=("accedio_mes", "size"), acc=("accedio_mes", "sum")
        ).reset_index()
        g["mes"] = mes
        g["tasa_%"] = (100 * g["acc"] / g["n"]).round(1)
        filas.append(g)
    mensual = pd.concat(filas)
    print(mensual.pivot_table(index=["poblacion", "mes"], columns="anio",
                              values="tasa_%").to_string())

    # ============================================================
    print("\n" + "=" * 78)
    print("4. ¿LA CONCLUSIÓN DEPENDE DEL UMBRAL? (robustez)")
    print("=" * 78)
    print("Si con 1, 5 y 10 días la conclusión es la misma, es una conclusión sólida.\n")
    for u in [1, 5, 10]:
        tmp = df.copy()
        tmp["accedio"] = tmp["dias_total"] >= u
        t = tasa(tmp, ["poblacion", "anio"])
        p = t.pivot_table(index="poblacion", columns="anio", values="tasa_%")
        p["var_pp"] = (p[2026] - p[2025]).round(1)
        print(f"--- umbral >= {u} día(s) ---")
        print(p.to_string(), "\n")


if __name__ == "__main__":
    main()