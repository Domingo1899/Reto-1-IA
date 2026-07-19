"""
ciclo_estudiantes.py — ¿La caída se concentra en algún ciclo?

QUÉ RESPONDE
De todos los cortes que probamos (sexo, rubro, zona, departamento), los únicos
con señal real fueron la INTENSIDAD de uso y el quintil. Este script agrega el
corte por CICLO, con las tres precauciones que aprendimos por las malas:

  1. UMBRALES MÚLTIPLES (1, 5, 10 días). La caída de acceso (>=1 día) es chica;
     la de uso sostenido (>=10) es grande. Si mirás un solo umbral, te perdés
     el hallazgo. Además calculamos la PROFUNDIZACIÓN (var_pp@10 - var_pp@1):
     cuánto más cae el uso sostenido que el acceso mínimo, por ciclo.

  2. CAMBIO RELATIVO además de pp. Un ciclo que parte de 20% no puede caer
     9 pp aunque colapse; uno que parte de 95% sí. Comparar pp entre ciclos
     con bases distintas es el mismo efecto piso que nos confundió en los
     quintiles. Por eso va var_rel_% = (t2026 - t2025) / t2025 * 100.

  3. FILTRO DE RUIDO. La columna ciclo tiene categorías con n=1, n=4, n=11 que
     producen variaciones de +2800% sin significar nada. Y hay ciclos que
     existen solo en un año (la reforma renombró categorías): esos NO son una
     caída, son un cambio de nomenclatura, y compararlos es un error. Se
     separan en una tabla aparte en vez de mezclarlos en el ranking.

SALIDA
  - Por consola: ranking de ciclos comparables + tabla de no comparables.
  - CSV: data/processed/metricas/ciclo_estudiantes.csv
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")

POBLACION = "estudiantes"
MESES = ["Dias4", "Dias5", "Dias6"]
UMBRALES = [1, 5, 10]

# n mínimo en AMBOS años para considerar un ciclo comparable.
# Con ~500k estudiantes, por debajo de 500 el % es puro ruido muestral.
MIN_N = 500

# Mismo formato que el resto de la carpeta metricas/ (Windows/Power BI español).
# Si algún día volvés al CSV estándar, sacá estos dos parámetros acá y en
# cualquier script que lea de metricas/ (visualizaciones.py, por ejemplo).
CSV_KW = dict(index=False, sep=";", decimal=",")


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar() -> pd.DataFrame:
    """Carga 2025+2026 de la población, normalizada y con dias_total."""
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{POBLACION}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]

        # Duplicados exactos: en docentes 2025 hay decenas de miles. Si no se
        # sacan, 2025 queda inflado y la caída se exagera.
        df = df.drop_duplicates()

        df["ciclo"] = df["ciclo"].map(norm)

        cols = [norm(c) for c in MESES]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dias_total"] = df[cols].sum(axis=1, min_count=1)

        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def tasa_por_ciclo(df: pd.DataFrame, umbral: int) -> pd.DataFrame:
    """Tabla ciclo x año con n y % que alcanzó el umbral de días."""
    d = df.copy()
    d["ok"] = d["dias_total"] >= umbral
    g = d.groupby(["ciclo", "anio"], dropna=False).agg(
        n=("ok", "size"), acc=("ok", "sum")
    ).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)

    piv = g.pivot_table(index="ciclo", columns="anio", values=["n", "acc", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index()


def comparar(piv: pd.DataFrame, umbral: int) -> pd.DataFrame:
    """Agrega var_pp y var_rel_%. Devuelve todas las filas (sin filtrar)."""
    p = piv.copy()
    p["umbral"] = umbral
    p["var_pp"] = (p["tasa_2026"] - p["tasa_2025"]).round(1)
    # Cambio relativo: la lente correcta para comparar ciclos con bases
    # distintas (evita el efecto piso).
    p["var_rel_%"] = (100 * (p["tasa_2026"] - p["tasa_2025"]) / p["tasa_2025"]).round(1)
    return p


def main():
    pd.set_option("display.width", 200)
    print(f"Cargando {POBLACION}...")
    df = cargar()
    print(f"  {len(df):,} filas (2025+2026, sin duplicados exactos)\n")

    tablas = [comparar(tasa_por_ciclo(df, u), u) for u in UMBRALES]
    todo = pd.concat(tablas, ignore_index=True)

    # ---- separar comparables de no comparables -------------------------
    # Un ciclo es comparable si tiene MIN_N registros en LOS DOS años.
    # Los que aparecen/desaparecen son renombres de la reforma, no caídas.
    n25 = todo["n_2025"].fillna(0)
    n26 = todo["n_2026"].fillna(0)
    ok = (n25 >= MIN_N) & (n26 >= MIN_N)

    comparables = todo[ok].copy()
    descartados = todo[~ok & (todo["umbral"] == 1)].copy()

    print("=" * 78)
    print(f"CICLOS NO COMPARABLES (n < {MIN_N} en algún año) — NO usar en conclusiones")
    print("=" * 78)
    if descartados.empty:
        print("  (ninguno)")
    else:
        d = descartados[["ciclo", "n_2025", "n_2026", "tasa_2025", "tasa_2026"]]
        d = d.fillna(0).sort_values("n_2026", ascending=False)
        print(d.to_string(index=False))
        print("\n  Ojo: los que van de n>0 a n=0 (o al revés) son categorías que la")
        print("  reforma renombró. La 'caída' de -100% es de nomenclatura, no de uso.")

    # ---- ranking por umbral --------------------------------------------
    for u in UMBRALES:
        sub = comparables[comparables["umbral"] == u].sort_values("var_pp")
        print("\n" + "=" * 78)
        print(f"CAÍDA POR CICLO — umbral >= {u} día(s)   [{POBLACION}]")
        print("=" * 78)
        print(sub[["ciclo", "n_2025", "n_2026", "tasa_2025", "tasa_2026",
                   "var_pp", "var_rel_%"]].to_string(index=False))

    # ---- profundización: el puente con la narrativa central -------------
    # var_pp@10 - var_pp@1. Muy negativo = en ese ciclo la caída NO es de
    # acceso sino de intensidad (la gente sigue entrando, pero mucho menos).
    a1 = comparables[comparables["umbral"] == 1].set_index("ciclo")["var_pp"]
    a10 = comparables[comparables["umbral"] == 10].set_index("ciclo")["var_pp"]
    prof = pd.DataFrame({"var_pp_umbral1": a1, "var_pp_umbral10": a10}).dropna()
    prof["profundizacion"] = (prof["var_pp_umbral10"] - prof["var_pp_umbral1"]).round(1)
    prof = prof.sort_values("profundizacion").reset_index()

    print("\n" + "=" * 78)
    print("PROFUNDIZACIÓN: ¿en qué ciclo la caída es de INTENSIDAD y no de acceso?")
    print("=" * 78)
    print("(profundizacion = var_pp@10 - var_pp@1; más negativo = más específico")
    print(" de la intensidad. Es el mismo efecto que vimos a nivel país.)\n")
    print(prof.to_string(index=False))

    # ---- guardar --------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["ciclo", "umbral", "n_2025", "n_2026", "acc_2025", "acc_2026",
            "tasa_2025", "tasa_2026", "var_pp", "var_rel_%"]
    ruta = OUT / f"ciclo_{POBLACION}.csv"
    comparables[cols].sort_values(["umbral", "var_pp"]).to_csv(ruta, **CSV_KW)
    print(f"\n  -> guardado {ruta}")

    ruta_prof = OUT / f"ciclo_profundizacion_{POBLACION}.csv"
    prof.to_csv(ruta_prof, **CSV_KW)
    print(f"  -> guardado {ruta_prof}")


if __name__ == "__main__":
    main()