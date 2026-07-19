"""
reforma.py — ¿La caída de intensidad se explica por la Transformación Educativa?

LO QUE ESTE SCRIPT PUEDE Y NO PUEDE HACER
No puede probar causalidad. Tenemos dos fotos (2025 y 2026) y la reforma ocurre
entre las dos: cualquier cosa que haya pasado en 2026 está perfectamente
confundida con la reforma. Con este diseño, "cayó todo y la reforma fue en 2026"
no es evidencia, es una coincidencia temporal con n=1.

Lo que SÍ puede hacer es someter la hipótesis a pruebas que podrían refutarla.
La lógica es la de EXPOSICIÓN DIFERENCIAL:

    Si la reforma causó la caída, los grupos MÁS expuestos a la reforma deben
    caer MÁS que los menos expuestos. Si caen todos igual, la reforma no lo
    explica y hay que buscar una causa común al sistema entero.

Un resultado negativo acá es tan valioso como uno positivo: descarta una
hipótesis, que es exactamente lo que Ceibal necesita para no invertir en la
dirección equivocada.

QUÉ HACE, EN ORDEN
  1. EVIDENCIA ESTRUCTURAL — cuánto se movieron las categorías entre años.
     Mide el tamaño del recableado que hizo la reforma en los propios datos.
  2. CONTAMINACIÓN POR RECLASIFICACIÓN — usando el ID de persona, cuántos
     alumnos cambiaron de ciclo entre 2025 y 2026. Si es alto, TODAS nuestras
     comparaciones por ciclo están comparando poblaciones distintas y hay que
     decirlo antes de mostrarlas.
  3. EXPOSICIÓN DIFERENCIAL (el test principal) — compara la caída de los
     grupos expuestos contra los no expuestos.
  4. DOSIS-RESPUESTA — si hay grados de exposición, ¿la caída escala con ella?
  5. PLACEBO TEMPORAL — el patrón mes a mes. Un efecto de reforma debería
     notarse desde el inicio del año lectivo; un efecto de calendario (paros,
     feriados, Semana de Turismo cayendo en meses distintos) se concentra en
     meses puntuales.

ANTES DE CORRERLO: HAY QUE COMPLETAR EXPOSICION_2026
El script no puede saber qué ciclos alcanzó la reforma en 2026 — eso sale del
calendario oficial de implementación de ANEP, no de los datos. Buscá el
cronograma y completá el diccionario de abajo. Ese dato es el que convierte
este análisis en un test real; sin él, el script solo corre los pasos 1, 2 y 5.
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")

MESES = {"Dias4": "abril", "Dias5": "mayo", "Dias6": "junio"}
UMBRAL = 10  # intensidad: es donde vive la señal
CSV_KW = dict(index=False, sep=";", decimal=",")

# ---------------------------------------------------------------------------
# COMPLETAR: nivel de exposición a la reforma en 2026, por ciclo.
#
#   "alta"    -> el ciclo se transformó en 2026 (plan nuevo, nueva estructura)
#   "parcial" -> alcanzado en parte (algunos grados, implementación gradual)
#   "nula"    -> sin cambios en 2026 (ESTE ES EL GRUPO DE CONTROL, el más
#                importante de todos: sin control no hay test)
#
# Las claves son los valores de la columna 'ciclo' ya normalizados (minúsculas,
# sin tildes), tal como aparecen en la salida de ciclo_estudiantes.py.
#
# Los valores de abajo están en None a propósito: NO los inventé. Completalos
# con el cronograma oficial. Si dejás alguno en None, ese ciclo queda fuera
# del test (no se fuerza a ningún grupo).
# ---------------------------------------------------------------------------
EXPOSICION_2026 = {
    "3er. ciclo ebi": None,
    "4to. ciclo": None,
    "educacion basica integrada": None,
    "primaria": None,
    "primaria especial": None,
    "ciclo basico": None,
    "bachillerato": None,
    "bachillerato tecnologico": None,
    "bachillerato tecnico profesional": None,
    "bachillerato figari": None,
    "formacion profesional basica": None,
    "educacion media tecnologica": None,
    "educacion media rural": None,
}

PISTAS_ID = ["id_persona", "id_alumno", "id_docente", "id_estudiante", "documento", "id"]


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def detectar_id(df: pd.DataFrame):
    """Busca la columna identificadora de persona (la de mayor cardinalidad)."""
    cands = [c for c in df.columns
             if any(c == p or c.startswith(p) for p in PISTAS_ID)
             and "centro" not in c]
    if not cands:
        return None
    cands.sort(key=lambda c: df[c].nunique(), reverse=True)
    return cands[0]


def cargar(pob: str) -> pd.DataFrame:
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{pob}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()
        if "ciclo" in df.columns:
            df["ciclo"] = df["ciclo"].map(norm)
        for c in MESES:
            df[norm(c)] = pd.to_numeric(df[norm(c)], errors="coerce")
        df["dias_total"] = df[[norm(c) for c in MESES]].sum(axis=1, min_count=1)
        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def tasa(df, por, umbral=UMBRAL):
    d = df.copy()
    d["ok"] = d["dias_total"] >= umbral
    g = d.groupby(por, dropna=False).agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
    return g


# ============================================================
# 1. EVIDENCIA ESTRUCTURAL
# ============================================================
def paso_estructura(df: pd.DataFrame):
    print("\n" + "=" * 74)
    print("1. EVIDENCIA ESTRUCTURAL: ¿cuánto movió la reforma las categorías?")
    print("=" * 74)

    g = df.groupby(["ciclo", "anio"]).size().unstack(fill_value=0)
    g.columns = [f"n_{c}" for c in g.columns]
    g["var_%"] = (100 * (g["n_2026"] - g["n_2025"]) /
                  g["n_2025"].where(g["n_2025"] > 0)).round(1)

    nuevos = g[g["n_2025"] == 0]
    muertos = g[g["n_2026"] == 0]
    movidos = g[(g["n_2025"] > 0) & (g["n_2026"] > 0) & (g["var_%"].abs() > 25)]

    print(f"\n  ciclos que aparecen solo en 2026 : {len(nuevos)}")
    print(f"  ciclos que desaparecen en 2026   : {len(muertos)}")
    print(f"  ciclos que cambian >25% de tamaño: {len(movidos)}")

    reclasificados = int(nuevos["n_2026"].sum() + muertos["n_2025"].sum())
    print(f"\n  personas en categorías que nacen o mueren: {reclasificados:,} "
          f"({100*reclasificados/len(df):.1f}% del total)")
    print("\n  Esto NO mide el efecto de la reforma sobre el uso. Mide que la")
    print("  reforma efectivamente reorganizó la estructura entre las dos fotos,")
    print("  que es la precondición para que la hipótesis sea siquiera plausible.")

    if len(movidos):
        print("\n  --- ciclos con mayor recableado ---")
        print(movidos.sort_values("var_%").to_string())


# ============================================================
# 2. CONTAMINACIÓN POR RECLASIFICACIÓN
# ============================================================
def paso_reclasificacion(df: pd.DataFrame, col_id: str):
    print("\n" + "=" * 74)
    print("2. CONTAMINACIÓN: ¿los mismos alumnos cambiaron de ciclo entre años?")
    print("=" * 74)

    if col_id is None:
        print("  No se detectó columna de ID de persona. Paso omitido.")
        return

    piv = df.pivot_table(index=col_id, columns="anio", values="ciclo",
                         aggfunc="first")
    piv = piv.dropna()
    if piv.empty:
        print("  No hay personas presentes en ambos años. Paso omitido.")
        return

    cambio = (piv[2025] != piv[2026])
    print(f"  personas presentes en ambos años : {len(piv):,}")
    print(f"  cambiaron de ciclo               : {cambio.sum():,} ({100*cambio.mean():.1f}%)")
    print("\n  OJO: parte de ese cambio es NATURAL (un alumno promueve de grado y")
    print("  puede cambiar de ciclo). Lo relevante es el exceso sobre lo esperable.")
    print("  Si el % es muy alto, las comparaciones 'mismo ciclo 2025 vs 2026'")
    print("  están comparando poblaciones distintas, y eso hay que aclararlo")
    print("  antes de mostrar el análisis por ciclo.")

    trans = (piv.groupby([2025, 2026]).size()
             .reset_index(name="n").sort_values("n", ascending=False))
    trans = trans[trans[2025] != trans[2026]]
    print("\n  --- 12 transiciones de ciclo más frecuentes ---")
    print(trans.head(12).to_string(index=False))


# ============================================================
# 3. EXPOSICIÓN DIFERENCIAL (test principal)
# ============================================================
def paso_exposicion(df: pd.DataFrame):
    print("\n" + "=" * 74)
    print("3. TEST PRINCIPAL: ¿caen más los ciclos expuestos a la reforma?")
    print("=" * 74)

    mapa = {k: v for k, v in EXPOSICION_2026.items() if v is not None}
    if not mapa:
        print("\n  EXPOSICION_2026 está sin completar, así que este test no corre.")
        print("  Es el paso que convierte esto en un análisis y no en una")
        print("  coincidencia temporal. Necesitás el cronograma oficial de")
        print("  implementación de la reforma para llenarlo.")
        print("\n  Sin grupo de control ('nula'), la hipótesis no es refutable:")
        print("  cualquier resultado sería compatible con 'fue la reforma'.")
        return None

    d = df[df["ciclo"].isin(mapa)].copy()
    d["exposicion"] = d["ciclo"].map(mapa)

    g = tasa(d, ["exposicion", "anio"])
    piv = g.pivot_table(index="exposicion", columns="anio", values=["n", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["var_pp"] = (piv["tasa_2026"] - piv["tasa_2025"]).round(1)
    piv = piv.reset_index()

    print()
    print(piv.to_string(index=False))

    if {"alta", "nula"} <= set(piv["exposicion"]):
        alta = piv.loc[piv["exposicion"] == "alta", "var_pp"].iloc[0]
        nula = piv.loc[piv["exposicion"] == "nula", "var_pp"].iloc[0]
        did = round(alta - nula, 1)
        print(f"\n  DIFERENCIA EN DIFERENCIAS: {alta} - ({nula}) = {did} pp")
        print("\n  CÓMO LEERLO:")
        if did < -2:
            print("  Los expuestos caen sustancialmente más que el control.")
            print("  -> COMPATIBLE con que la reforma explique parte de la caída.")
            print("     Sigue sin ser prueba causal: puede haber otras diferencias")
            print("     entre los grupos además de la exposición.")
        elif did > 2:
            print("  Los expuestos caen MENOS que el control. Va en contra de la")
            print("  hipótesis: habría que revisarla o descartarla.")
        else:
            print("  Expuestos y no expuestos caen prácticamente igual.")
            print("  -> La reforma NO explica la caída. Buscá una causa común a")
            print("     todo el sistema (calendario, plataforma, conectividad).")
    else:
        print("\n  Falta el grupo 'alta' o el grupo 'nula': sin ambos no hay")
        print("  comparación posible.")
    return piv


# ============================================================
# 4. DOSIS-RESPUESTA
# ============================================================
def paso_dosis(piv):
    if piv is None:
        return
    print("\n" + "=" * 74)
    print("4. DOSIS-RESPUESTA: ¿a más exposición, más caída?")
    print("=" * 74)
    orden = ["nula", "parcial", "alta"]
    sub = piv[piv["exposicion"].isin(orden)].copy()
    if len(sub) < 3:
        print("  Hacen falta los tres niveles para evaluar la gradiente.")
        return
    sub["_o"] = sub["exposicion"].map({v: i for i, v in enumerate(orden)})
    sub = sub.sort_values("_o")
    print()
    print(sub[["exposicion", "n_2026", "tasa_2025", "tasa_2026", "var_pp"]].to_string(index=False))
    caidas = sub["var_pp"].tolist()
    monotona = all(caidas[i] >= caidas[i + 1] for i in range(len(caidas) - 1))
    print(f"\n  gradiente monótona (nula -> parcial -> alta): "
          f"{'SÍ' if monotona else 'NO'}")
    print("  Una gradiente ordenada es evidencia bastante más fuerte que una")
    print("  simple diferencia entre dos grupos: es difícil de explicar por azar.")


# ============================================================
# 5. PLACEBO TEMPORAL
# ============================================================
def paso_meses(df: pd.DataFrame):
    print("\n" + "=" * 74)
    print("5. PLACEBO TEMPORAL: ¿el patrón mensual parece reforma o calendario?")
    print("=" * 74)

    filas = []
    for col, mes in MESES.items():
        c = norm(col)
        d = df.copy()
        d["ok"] = d[c] >= 1  # al menos un día en ese mes
        g = d.groupby("anio").agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
        g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
        filas.append({"mes": mes,
                      "tasa_2025": g.loc[g["anio"] == 2025, "tasa"].iloc[0],
                      "tasa_2026": g.loc[g["anio"] == 2026, "tasa"].iloc[0]})
    m = pd.DataFrame(filas)
    m["var_pp"] = (m["tasa_2026"] - m["tasa_2025"]).round(1)
    print()
    print(m.to_string(index=False))

    rango = m["var_pp"].max() - m["var_pp"].min()
    print("\n  CÓMO LEERLO:")
    print(f"  dispersión entre meses: {rango:.1f} pp")
    if rango > 3:
        print("  La caída se concentra en meses puntuales -> sospechá de un")
        print("  efecto de CALENDARIO (paros, feriados, Semana de Turismo cayendo")
        print("  en meses distintos entre 2025 y 2026) antes que de la reforma.")
        print("  Este confusor es barato de descartar y caro de ignorar:")
        print("  conviene pedirle a Ceibal el calendario lectivo de ambos años.")
    else:
        print("  La caída es pareja entre meses -> compatible con un cambio")
        print("  estructural sostenido, no con un evento puntual de calendario.")


def main():
    pd.set_option("display.width", 200)

    for pob in ("estudiantes", "docentes"):
        print("\n\n" + "#" * 74)
        print(f"#  HIPÓTESIS REFORMA — {pob.upper()}")
        print("#" * 74)

        df = cargar(pob)
        if "ciclo" not in df.columns:
            print("  No hay columna 'ciclo'. Se omite esta población.")
            continue

        col_id = detectar_id(df)
        print(f"  filas: {len(df):,} | columna de ID detectada: {col_id}")

        paso_estructura(df)
        paso_reclasificacion(df, col_id)
        piv = paso_exposicion(df)
        paso_dosis(piv)
        paso_meses(df)

    print("\n" + "=" * 74)
    print("QUÉ HACER CON ESTO")
    print("=" * 74)
    print("""
  El resultado honesto que se puede llevar a Ceibal tiene tres partes:

    1. La caída es de intensidad y es transversal a todo el sistema
       (la mediana de los centros cae casi lo mismo que el país).
    2. La reforma es una hipótesis PLAUSIBLE porque coincide en el tiempo
       y porque reorganizó la estructura (paso 1 lo cuantifica).
    3. Con estos datos no se puede atribuir causalmente. Para hacerlo haría
       falta: el cronograma de implementación por grado, el calendario
       lectivo de ambos años, y saber si Ceibal cambió algo en CREA o
       lanzó herramientas alternativas en 2026.

  Presentar el punto 3 no debilita el trabajo: muestra que sabés distinguir
  una correlación de una causa, que es exactamente lo que se espera de un
  análisis serio.
    """)


if __name__ == "__main__":
    main()