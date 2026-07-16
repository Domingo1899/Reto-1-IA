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
 
NOTA (cambio agregado): en TODAS las tablas donde aparece una cantidad de
personas (alumnos/docentes), esa cantidad se muestra en NÚMERO ABSOLUTO
y en PORCENTAJE lado a lado. Un % solo ("tasa_%") no te dice si estás
hablando de 5 personas o de 50.000; el número absoluto le da el contexto
al %.
"""
 
import unicodedata
from pathlib import Path
 
import pandas as pd
 
CARPETA = Path("data/interim/entrega-02-corregida")
 
# Carpeta donde vamos a guardar todas las tablas calculadas, en CSV,
# para que el dashboard (generar_dashboard.py) las pueda leer sin
# tener que recalcular nada.
OUT = Path("data/processed/metricas")
 
MESES = {"Dias4": "abril", "Dias5": "mayo", "Dias6": "junio"}
 
# Dimensiones por las que además del panorama general y por rubro,
# queremos desglosar (si la columna existe en los datos).
DIMENSIONES_EXTRA = ["sexo", "ciclo", "zona", "tipo_centro", "dept_nombre"]
 
# DECISIÓN (documentar en decisiones.md):
# "usuario que accedió" = al menos 1 día de acceso en el trimestre.
# Es la lectura más directa del PDF ("cantidad de usuarios que accedieron").
# Más abajo probamos también otros umbrales, para ver si la conclusión cambia.
UMBRAL = 1
 
# DECISIÓN (consultada con el profe): analizar por 2 bloques de la reforma
#   - EBI  (Educación Básica Integrada): Inicial 3-5 años + Primaria 1º-6º
#          + Ciclo Básico 7º-9º
#   - EMS  (Educación Media Superior): Bachillerato 1º-3º
# Los valores de abajo son los que aparecen REALMENTE en la columna "ciclo"
# de docentes/estudiantes 2025/2026 (ya en minúsculas y sin tildes, como
# los deja norm()). Categorías que no encajan claramente en ninguno de los
# dos bloques (4to. ciclo, educación media rural, FPB, CTT, educación media
# profesional, etc.) quedan FUERA a propósito -> ciclo_a_bloque() devuelve
# None para ellas, y esos registros se excluyen del análisis por bloque
# (no se fuerzan a un bloque para no ensuciar el análisis).
EBI_VALORES = {
    "primaria",
    "primaria especial",
    "primaria especial - disc. auditiva",
    "educacion basica integrada",
    "3er. ciclo ebi",
    "ciclo basico",
    "ciclo basico tecnologico",
    "articulacion educacion media basica",
    "educacion media basica tecnologica",
}
 
EMS_VALORES = {
    "bachillerato",
    "bachillerato figari",
    "bachillerato profesional",
    "bachillerato profesional trayectos",
    "bachillerato tecnico profesional",
    "bachillerato tecnologico",
    "educacion media superior",
    "educacion media tecnologica",
    "educacion media tecnologica finest",
}
 
 
def ciclo_a_bloque(ciclo_norm: str):
    """Devuelve 'ebi', 'ems', o None si el ciclo no fue clasificado
    (categoría dudosa que decidimos dejar fuera del análisis por bloque)."""
    if ciclo_norm in EBI_VALORES:
        return "ebi"
    if ciclo_norm in EMS_VALORES:
        return "ems"
    return None
 
 
def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))
 
 
def guardar(df: pd.DataFrame, nombre: str) -> None:
    """Guarda un DataFrame en data/processed/metricas/<nombre>.csv"""
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / f"{nombre}.csv"
    df.to_csv(ruta, index=False)
    print(f"  -> guardado {ruta}")
 
 
def preparar(df: pd.DataFrame, poblacion: str, anio: int) -> pd.DataFrame:
    """Normaliza y calcula las columnas derivadas que necesitamos."""
    df = df.copy()
 
    # 1. Nombres de columna a minúsculas -> resuelve Zona/ZONA, IvsMedia/IVSMEDIA
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]
 
    # 2. Valores de texto a minúsculas -> resuelve 'Urbana' vs 'URBANA'
    for col in ["zona", "tipo_centro", "rubro", "dept_nombre", "ciclo", "sexo"]:
        if col in df.columns:
            df[col] = df[col].map(norm)
 
    # 2b. Bloque EBI/EMS a partir de "ciclo" (ya normalizado arriba).
    #     Las categorías dudosas quedan en None -> se filtran antes de
    #     armar la tabla por bloque (ver main()).
    if "ciclo" in df.columns:
        df["bloque"] = df["ciclo"].map(ciclo_a_bloque)
 
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
    """Compara 2025 vs 2026 en la dimensión que le pases.
 
    Cada cantidad de personas sale en número Y en % en la misma tabla:
      - n_registrados_20XX -> cuánta gente hay en la base (número)
      - var_registrados_%  -> cuánto cambió esa base (%)
      - n_accedieron_20XX  -> cuánta gente accedió, en NÚMERO absoluto
      - tasa_%_20XX        -> qué % de la base accedió
      - var_tasa_pp        -> cuánto cambió esa tasa, en puntos porcentuales
    """
    por = ["poblacion", "anio"] + ([dimension] if dimension else [])
    t = tasa(df, por)
 
    idx = ["poblacion"] + ([dimension] if dimension else [])
    piv = t.pivot_table(
        index=idx, columns="anio",
        values=["n_registrados", "n_accedieron", "tasa_%"]
    ).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
 
    # n_registrados y n_accedieron son conteos -> a entero (evita "82.0")
    for c in ("n_registrados_2025", "n_registrados_2026",
              "n_accedieron_2025", "n_accedieron_2026"):
        piv[c] = piv[c].fillna(0).astype(int)
 
    piv["var_registrados_%"] = (
        100 * (piv["n_registrados_2026"] - piv["n_registrados_2025"])
        / piv["n_registrados_2025"]
    ).round(1)
    piv["var_tasa_pp"] = (piv["tasa_%_2026"] - piv["tasa_%_2025"]).round(1)
 
    return piv[[*idx,
                "n_registrados_2025", "n_registrados_2026", "var_registrados_%",
                "n_accedieron_2025", "n_accedieron_2026",
                "tasa_%_2025", "tasa_%_2026", "var_tasa_pp"]]
 
 
def mensual_por(df, dimension):
    """Evolución mensual de la tasa de acceso, desglosada por una dimensión.
    Pensado para armar heatmaps (dimensión x mes) en el dashboard.
 
    Igual que en comparar(): la cantidad de gente que accedió sale en
    número (acc_20XX) y en % (tasa_%_20XX), no solo en %.
    """
    filas = []
    for col, mes in MESES.items():
        c = norm(col)
        tmp = df.copy()
        tmp["accedio_mes"] = tmp[c] >= UMBRAL
        g = tmp.groupby(["poblacion", "anio", dimension], dropna=False).agg(
            n=("accedio_mes", "size"), acc=("accedio_mes", "sum")
        ).reset_index()
        g["mes"] = mes
        g["tasa_%"] = (100 * g["acc"] / g["n"]).round(1)
        filas.append(g)
    mensual = pd.concat(filas, ignore_index=True)
 
    piv = mensual.pivot_table(
        index=["poblacion", dimension, "mes"],
        columns="anio", values=["n", "acc", "tasa_%"]
    ).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
 
    piv = piv.rename(columns={
        "n_2025": "n_registrados_2025", "n_2026": "n_registrados_2026",
        "acc_2025": "n_accedieron_2025", "acc_2026": "n_accedieron_2026",
    })
 
    for c in ("n_registrados_2025", "n_registrados_2026",
              "n_accedieron_2025", "n_accedieron_2026"):
        piv[c] = piv[c].fillna(0).astype(int)
 
    piv["var_tasa_pp"] = (piv["tasa_%_2026"] - piv["tasa_%_2025"]).round(1)
 
    return piv[["poblacion", dimension, "mes",
                "n_registrados_2025", "n_registrados_2026",
                "n_accedieron_2025", "n_accedieron_2026",
                "tasa_%_2025", "tasa_%_2026", "var_tasa_pp"]]
 
 
def main():
    pd.set_option("display.width", 200)
 
    print("Cargando y preparando...")
    partes = []
    for ruta in sorted(CARPETA.glob("*.parquet")):
        poblacion, anio = ruta.stem.split("_")
        partes.append(preparar(pd.read_parquet(ruta), poblacion, int(anio)))
    df = pd.concat(partes, ignore_index=True)
 
    dims_presentes = [d for d in DIMENSIONES_EXTRA if d in df.columns]
    print(f"\nDimensiones extra encontradas en los datos: {dims_presentes}")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("1. LA PREGUNTA CENTRAL: ¿menos gente, o menos uso?")
    print("=" * 78)
    general = comparar(df)
    print(general.to_string(index=False))
    print("""
  CÓMO LEERLO:
    var_registrados_% -> cambió la cantidad de gente en la base (denominador)
    n_accedieron_20XX -> cuánta gente entró a CREA, en número absoluto
    var_tasa_pp       -> cambió el % que entró a CREA (comportamiento)
 
    Si var_tasa_pp ~ 0  -> la caída es de POBLACIÓN, no de uso.
    Si var_tasa_pp < 0  -> la gente que está, entra menos. Eso sí es CREA.
    """)
    guardar(general, "comparar_general")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("2. POR SUBSISTEMA (Rubro)   DGEIP=Primaria DGES=Secundaria DGETP=UTU")
    print("=" * 78)
    rubro = comparar(df, "rubro")
    print(rubro.to_string(index=False))
    guardar(rubro, "comparar_rubro")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("3. OTRAS DIMENSIONES (sexo, ciclo, zona, tipo de centro, departamento)")
    print("=" * 78)
    for dim in dims_presentes:
        tabla = comparar(df, dim)
        print(f"\n--- por {dim} ---")
        print(tabla.to_string(index=False))
        guardar(tabla, f"comparar_{dim}")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("3b. POR BLOQUE (EBI vs EMS, post-reforma)")
    print("=" * 78)
    if "bloque" in df.columns:
        df_bloque = df[df["bloque"].notna()]
        excluidas = len(df) - len(df_bloque)
        print(f"(se excluyen {excluidas:,} registros con ciclo no clasificado "
              f"en EBI/EMS -> {100*excluidas/len(df):.1f}% del total)")
        bloque = comparar(df_bloque, "bloque")
        print(bloque.to_string(index=False))
        guardar(bloque, "comparar_bloque")
    else:
        print("No se encontró la columna 'ciclo', se salta este bloque.")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("4. EVOLUCIÓN MES A MES (¿la caída se concentra en algún mes?)")
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
    mensual = pd.concat(filas, ignore_index=True)
    print(mensual.pivot_table(index=["poblacion", "mes"], columns="anio",
                              values="tasa_%").to_string())
    guardar(mensual, "evolucion_mensual")
 
    # Heatmaps: evolución mensual desglosada por departamento y por rubro
    if "dept_nombre" in df.columns:
        guardar(mensual_por(df, "dept_nombre"), "mensual_por_dept_nombre")
    guardar(mensual_por(df, "rubro"), "mensual_por_rubro")
 
    # ============================================================
    print("\n" + "=" * 78)
    print("5. ¿LA CONCLUSIÓN DEPENDE DEL UMBRAL? (robustez)")
    print("=" * 78)
    print("Si con 1, 5 y 10 días la conclusión es la misma, es una conclusión sólida.\n")
    robustez_filas = []
    for u in [1, 5, 10]:
        tmp = df.copy()
        tmp["accedio"] = tmp["dias_total"] >= u
        t = tasa(tmp, ["poblacion", "anio"])
        p = t.pivot_table(index="poblacion", columns="anio",
                           values=["n_registrados", "n_accedieron", "tasa_%"])
        p.columns = [f"{a}_{b}" for a, b in p.columns]
        p["var_pp"] = (p["tasa_%_2026"] - p["tasa_%_2025"]).round(1)
        p = p.reset_index()
        p["umbral"] = u
        robustez_filas.append(p)
        print(f"--- umbral >= {u} día(s) ---")
        print(p.to_string(), "\n")
    robustez = pd.concat(robustez_filas, ignore_index=True)
    guardar(robustez, "robustez_umbral")
 
    print("\nListo. Todas las tablas quedaron guardadas en:", OUT.resolve())
 
 
if __name__ == "__main__":
    main()