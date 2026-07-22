"""
Reto-1-IA · CHEQUEO DE PANEL
=============================================================================
Responde dos preguntas:

  A) La caida de INTENSIDAD (-1.7 / -4.7 / -6.8 pp) ¿es comportamiento
     o es recambio de matricula?

  B) El ARRASTRE DOCENTE ¿sigue existiendo si miro solo a las mismas
     personas en los dos anios?

Todo lo que tenes que tocar esta aca abajo, en CONFIG.
=============================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG  <-- AJUSTAR ESTO
# =============================================================================
BASE = Path("/Users/santiagoalba/Desktop/Reto-1-IA")
INTERIM = BASE / "data/interim/entrega-02-corregida"

EST_2025 = INTERIM / "estudiantes_2025.parquet"
EST_2026 = INTERIM / "estudiantes_2026.parquet"
DOC_2025 = INTERIM / "docentes_2025.parquet"
DOC_2026 = INTERIM / "docentes_2026.parquet"

OUT = BASE / "data/processed/metricas"
CSV_KW = dict(index=False, sep=";", decimal=",", encoding="utf-8-sig")
UMBRALES = [1, 5, 10]
# Nombres posibles de cada columna. Prueba en orden y usa el primero que exista.
CANDIDATOS = {
    "id":     ["ID_persona", "id", "ID", "id_persona"],
    "centro": ["ID_CENTRO", "centro", "CENTRO", "id_centro"],
}
MESES = ["Dias4", "Dias5", "Dias6"]   # abril, mayo, junio
def preparar(df, etiqueta):
    """Una fila por persona, con el total de dias de abril+mayo+junio."""
    i = col(df, "id", etiqueta)
    c = col(df, "centro", etiqueta)

    faltan = [m for m in MESES if m not in df.columns]
    if faltan:
        print(f"  ERROR: faltan columnas de meses en {etiqueta}: {faltan}")
        print(f"  Disponibles: {list(df.columns)}")
        sys.exit(1)

    d = df[[i, c] + MESES].copy()

    # los meses vienen como texto: pasar a numero
    for m in MESES:
        d[m] = pd.to_numeric(
            d[m].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
    d[MESES] = d[MESES].fillna(0)

    d["dias"] = d[MESES].sum(axis=1)

    print(f"  {etiqueta}: dias min={d['dias'].min():.0f} "
          f"max={d['dias'].max():.0f} media={d['dias'].mean():.1f}")

    n_filas = len(d)
    d = d.groupby(i, as_index=False).agg(
        centro=(c, "first"),
        dias=("dias", "max"),
    ).rename(columns={i: "id"})

    if n_filas != len(d):
        print(f"  {etiqueta}: {n_filas:,} filas -> {len(d):,} personas unicas")

    return d
# =============================================================================
# UTILIDADES
# =============================================================================
def col(df, clave, nombre_df):
    """Encuentra el nombre real de una columna probando candidatos."""
    for c in CANDIDATOS[clave]:
        if c in df.columns:
            return c
    print(f"\n  ERROR: no encuentro la columna '{clave}' en {nombre_df}.")
    print(f"  Columnas disponibles: {list(df.columns)}")
    print(f"  Agregala a CANDIDATOS['{clave}'] arriba en el script.\n")
    sys.exit(1)


def tasa(serie_dias, umbral):
    """% de personas que usaron CREA al menos 'umbral' dias."""
    return (serie_dias >= umbral).mean() * 100


def cargar():
    print("Cargando...")
    e25 = pd.read_parquet(EST_2025)
    e26 = pd.read_parquet(EST_2026)
    print(f"  estudiantes 2025: {len(e25):,} | 2026: {len(e26):,}")
    return e25, e26


# =============================================================================
# CHEQUEO A · ¿La caida de intensidad es comportamiento o recambio?
# =============================================================================
def chequeo_a(e25, e26):
    print("\n" + "=" * 74)
    print("CHEQUEO A · INTENSIDAD: ¿comportamiento o recambio de matricula?")
    print("=" * 74)

    a = preparar(e25, "est2025")
    b = preparar(e26, "est2026")

    p = a[["id", "dias"]].rename(columns={"dias": "dias_2025"}).merge(
        b[["id", "dias"]].rename(columns={"dias": "dias_2026"}),
        on="id", how="inner",
    )
    e25, d25 = a, "dias"   # para que el resto del codigo siga funcionando
    e26, d26 = b, "dias"

    print(f"\n  transversal 2025 : {len(e25):,} personas")
    print(f"  transversal 2026 : {len(e26):,} personas")
    print(f"  PANEL (en ambos) : {len(p):,} personas")
    print(f"  ingresantes 2026 : {len(e26) - len(p):,}")
    print(f"  egresados        : {len(e25) - len(p):,}")

    filas = []
    for u in UMBRALES:
        t_cross_25 = tasa(e25[d25], u)
        t_cross_26 = tasa(e26[d26], u)
        t_pan_25 = tasa(p["dias_2025"], u)
        t_pan_26 = tasa(p["dias_2026"], u)
        filas.append({
            "umbral": u,
            "cross_2025": round(t_cross_25, 1),
            "cross_2026": round(t_cross_26, 1),
            "cross_var_pp": round(t_cross_26 - t_cross_25, 1),
            "panel_2025": round(t_pan_25, 1),
            "panel_2026": round(t_pan_26, 1),
            "panel_var_pp": round(t_pan_26 - t_pan_25, 1),
        })

    tab = pd.DataFrame(filas)
    tab["composicion_pp"] = (tab["cross_var_pp"] - tab["panel_var_pp"]).round(1)
    print("\n" + tab.to_string(index=False))

    # --- lectura automatica ---
    print("\n  LECTURA:")
    cross = tab["cross_var_pp"].tolist()
    panel = tab["panel_var_pp"].tolist()

    escalera_cross = cross[0] > cross[-1]   # se profundiza con el umbral
    escalera_panel = panel[0] > panel[-1]

    print(f"    transversal: {cross[0]} -> {cross[-1]} pp  "
          f"(se profundiza: {'SI' if escalera_cross else 'NO'})")
    print(f"    panel      : {panel[0]} -> {panel[-1]} pp  "
          f"(se profundiza: {'SI' if escalera_panel else 'NO'})")

    if escalera_panel and panel[-1] < -1:
        print("\n    -> EL PUNTO 1 AGUANTA. Las mismas personas bajaron la")
        print("       frecuencia de uso. Es comportamiento real, no recambio.")
    elif abs(panel[-1]) < 1:
        print("\n    -> EL PUNTO 1 SE CAE. En el panel casi no hay caida:")
        print("       la baja de intensidad tambien era recambio de matricula.")
    else:
        print("\n    -> RESULTADO MIXTO. Hay algo de comportamiento y algo de")
        print("       recambio. Mira la columna composicion_pp para ver cuanto")
        print("       aporta cada parte en cada umbral.")

    tab.to_csv(OUT / "panel_intensidad.csv", **CSV_KW)
    print(f"\n  -> {OUT / 'panel_intensidad.csv'}")
    return p


# =============================================================================
# CHEQUEO B · ¿El arrastre docente sobrevive al panel?
# =============================================================================
def chequeo_b(umbral=5):
    print("\n" + "=" * 74)
    print(f"CHEQUEO B · ARRASTRE DOCENTE en panel (umbral = {umbral} dias)")
    print("=" * 74)

    e25, e26 = pd.read_parquet(EST_2025), pd.read_parquet(EST_2026)
    d25, d26 = pd.read_parquet(DOC_2025), pd.read_parquet(DOC_2026)

    def panelizar(a, b, etiqueta):
        pa = preparar(a, etiqueta).rename(columns={"dias": "dias_25"})
        pb = preparar(b, etiqueta)[["id", "dias"]].rename(columns={"dias": "dias_26"})
        return pa.merge(pb, on="id", how="inner")

    pe = panelizar(e25, e26, "estudiantes")
    pd_ = panelizar(d25, d26, "docentes")
    print(f"\n  panel estudiantes: {len(pe):,} | panel docentes: {len(pd_):,}")

    def por_centro(p, pref):
        g = p.groupby("centro").apply(
            lambda x: pd.Series({
                f"n_{pref}": len(x),
                f"{pref}_2025": tasa(x["dias_25"], umbral),
                f"{pref}_2026": tasa(x["dias_26"], umbral),
            }), include_groups=False
        ).reset_index()
        g[f"d_{pref}"] = g[f"{pref}_2026"] - g[f"{pref}_2025"]
        return g

    ce, cd = por_centro(pe, "est"), por_centro(pd_, "doc")
    m = ce.merge(cd, on="centro", how="inner")

    # filtro minimo: centros con muestra suficiente en ambas poblaciones
    m = m[(m["n_est"] >= 50) & (m["n_doc"] >= 20)].copy()
    print(f"  centros analizables (>=20 alumnos y >=5 docentes en panel): {len(m):,}")

    if len(m) < 30:
        print("\n  Muy pocos centros. Baja los minimos o revisa los IDs.")
        return

    r_p = m["d_doc"].corr(m["d_est"])
    r_s = m["d_doc"].corr(m["d_est"], method="spearman")
    pend = np.polyfit(m["d_doc"], m["d_est"], 1)[0]

    print(f"\n  correlacion Pearson  : r = {r_p:.3f}")
    print(f"  correlacion Spearman : r = {r_s:.3f}")
    print(f"  pendiente            : por cada 10 pp que cae el docente,")
    print(f"                         el alumno cae {pend * 10:.1f} pp")

    m["q"] = pd.qcut(m["d_doc"].rank(method="first"), 5, labels=[
        "Q1 (mas cayo)", "Q2", "Q3", "Q4", "Q5 (menos cayo)"])
    dr = m.groupby("q", observed=True).agg(
        centros=("d_doc", "size"),
        d_doc=("d_doc", "mean"),
        d_est=("d_est", "mean"),
    ).round(1).reset_index()
    print("\n  --- dosis-respuesta (solo panel) ---")
    print(dr.to_string(index=False))

    print("\n  LECTURA:")
    vals = dr["d_est"].tolist()
    monotona = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    if r_p > 0.20 and monotona:
        print("    -> EL PUNTO 2 AGUANTA. La relacion se mantiene con las mismas")
        print(f"       personas (r = {r_p:.3f}) y la escalera sigue ordenada.")
    elif r_p > 0.20:
        print(f"    -> La correlacion se mantiene (r = {r_p:.3f}) pero la escalera")
        print("       tiene saltos. Reportar el r, no la dosis-respuesta.")
    else:
        print(f"    -> EL PUNTO 2 SE DEBILITA MUCHO (r = {r_p:.3f}). Buena parte")
        print("       del arrastre era composicion. Hay que reformularlo.")

    m.round(2).to_csv(OUT / "panel_arrastre.csv", **CSV_KW)
    print(f"\n  -> {OUT / 'panel_arrastre.csv'}")


# =============================================================================
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    e25, e26 = cargar()
    chequeo_a(e25, e26)
    chequeo_b(umbral=5)
    print("\n" + "=" * 74 + "\nListo.\n")