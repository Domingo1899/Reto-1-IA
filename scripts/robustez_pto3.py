"""
robustez_arrastra.py — ¿Aguanta el r = 0.355 del arrastre docente?

POR QUÉ HACE FALTA
El hallazgo (donde cae el docente, caen los alumnos) tiene un flanco: el SHOCK
COMÚN DE CENTRO. Si una escuela tuvo un problema en 2026 —cambio de dirección,
conectividad, un error de registro, un centro que directamente dejó de usar
CREA— caen docentes y alumnos a la vez sin que uno arrastre al otro. Un puñado
de centros así puede inflar toda la correlación.

Este script somete el resultado a siete especificaciones distintas. Si el r se
mantiene parecido en todas, el hallazgo es sólido y se puede defender de
cualquier objeción. Si se desarma en alguna, es mejor enterarse ahora.

LAS SIETE PRUEBAS
  A. Base                    el número original, como referencia
  B. Sin colas extremas      saca el 2.5% de cada punta en ambas variables
  C. Sin centros apagados    saca los que cambian más de 40 pp (encendido/apagado)
  D. Solo centros grandes    n alto = menos ruido de muestreo pequeño
  E. Ponderado por alumnos   que no manden los centros chicos
  F. Por estrato             ¿aguanta dentro de cada tipo de centro y zona?
  G. Permutación             baraja los datos: el r por azar debería ser ~0

La G es la más importante de todas: si al romper la correspondencia entre
centros el r se va a cero, entonces el 0.355 no es un artefacto de cómo están
construidas las variables.

Sin dependencias nuevas: la correlación ponderada y la permutación están
implementadas a mano para no necesitar scipy.
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")

MESES = ["Dias4", "Dias5", "Dias6"]
UMBRAL = 10
MIN_EST, MIN_DOC = 30, 5
GRANDE_EST, GRANDE_DOC = 100, 15
EXTREMO = 40          # pp: por encima de esto se considera apagado/encendido
N_PERMUTACIONES = 500
CSV_KW = dict(index=False, sep=";", decimal=",")

ATRIBUTOS = ["tipo_centro", "zona", "rubro", "dept_nombre"]


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar(pob: str) -> pd.DataFrame:
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{pob}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()
        for col in ATRIBUTOS:
            if col in df.columns:
                df[col] = df[col].map(norm)
        cols = [norm(c) for c in MESES]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dias_total"] = df[cols].sum(axis=1, min_count=1)
        df["ok"] = df["dias_total"] >= UMBRAL
        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def por_centro(df, prefijo, minimo):
    g = (df.groupby(["id_centro", "anio"])
         .agg(n=("ok", "size"), acc=("ok", "sum")).reset_index())
    g["tasa"] = 100 * g["acc"] / g["n"]
    piv = g.pivot_table(index="id_centro", columns="anio", values=["n", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna(subset=["tasa_2025", "tasa_2026"])
    piv = piv[(piv["n_2025"] >= minimo) & (piv["n_2026"] >= minimo)]
    piv = piv.rename(columns={
        "tasa_2025": f"{prefijo}_25", "tasa_2026": f"{prefijo}_26",
        "n_2025": f"n_{prefijo}_25", "n_2026": f"n_{prefijo}_26"})
    piv[f"d_{prefijo}"] = piv[f"{prefijo}_26"] - piv[f"{prefijo}_25"]
    return piv


def corr_pond(x, y, w):
    """Correlación de Pearson ponderada (implementada a mano, sin scipy)."""
    w = np.asarray(w, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)


def linea(nombre, n, r, extra=""):
    print(f"  {nombre:<34} n={n:>5,}   r = {r:>6.3f}   {extra}")


def main():
    pd.set_option("display.width", 200)

    print("Cargando...")
    est = cargar("estudiantes")
    doc = cargar("docentes")

    ce = por_centro(est, "est", MIN_EST)
    cd = por_centro(doc, "doc", MIN_DOC)
    c = ce.join(cd, how="inner")

    # atributos del centro (para el análisis por estrato)
    disponibles = [a for a in ATRIBUTOS if a in est.columns]
    if disponibles:
        attr = (est[est["anio"] == 2026]
                .groupby("id_centro")[disponibles].first())
        c = c.join(attr)
    c = c.reset_index()
    print(f"  centros analizables: {len(c):,}")
    print(f"  atributos disponibles: {disponibles}")

    r_base = c["d_doc"].corr(c["d_est"])

    print("\n" + "=" * 74)
    print("ROBUSTEZ DE LA CORRELACIÓN CAÍDA DOCENTE ↔ CAÍDA ESTUDIANTIL")
    print("=" * 74 + "\n")

    # A ---------------------------------------------------------------
    linea("A. Base (todos los centros)", len(c), r_base)

    # B --- sin colas extremas ----------------------------------------
    lo_d, hi_d = c["d_doc"].quantile([.025, .975])
    lo_e, hi_e = c["d_est"].quantile([.025, .975])
    b = c[(c["d_doc"].between(lo_d, hi_d)) & (c["d_est"].between(lo_e, hi_e))]
    linea("B. Sin el 2.5% de cada cola", len(b), b["d_doc"].corr(b["d_est"]))

    # C --- sin centros apagados/encendidos ---------------------------
    cc = c[(c["d_doc"].abs() <= EXTREMO) & (c["d_est"].abs() <= EXTREMO)]
    linea(f"C. Sin cambios de más de {EXTREMO} pp", len(cc),
          cc["d_doc"].corr(cc["d_est"]))

    # D --- solo centros grandes --------------------------------------
    gr = c[(c["n_est_26"] >= GRANDE_EST) & (c["n_doc_26"] >= GRANDE_DOC)]
    linea(f"D. Solo centros grandes (≥{GRANDE_EST} al., ≥{GRANDE_DOC} doc.)",
          len(gr), gr["d_doc"].corr(gr["d_est"]))

    # E --- ponderado por alumnos -------------------------------------
    linea("E. Ponderado por cantidad de alumnos", len(c),
          corr_pond(c["d_doc"], c["d_est"], c["n_est_26"]))

    # combinada: la especificación más exigente
    ex = cc[(cc["n_est_26"] >= GRANDE_EST) & (cc["n_doc_26"] >= GRANDE_DOC)]
    if len(ex) > 30:
        linea("   (B+C+D combinadas)", len(ex), ex["d_doc"].corr(ex["d_est"]),
              "<- la más exigente")

    # F --- por estrato ------------------------------------------------
    print("\n  F. Dentro de cada estrato (¿aguanta en todos lados?)")
    for col in disponibles:
        if col == "dept_nombre":
            continue  # demasiadas categorías para la consola
        print(f"\n     -- por {col} --")
        for val, sub in c.groupby(col):
            if len(sub) < 40 or not val:
                continue
            r = sub["d_doc"].corr(sub["d_est"])
            print(f"     {str(val)[:30]:<32} n={len(sub):>5,}   r = {r:>6.3f}")

    # G --- permutación ------------------------------------------------
    rng = np.random.default_rng(42)
    x = c["d_doc"].to_numpy()
    y = c["d_est"].to_numpy()
    rs = np.empty(N_PERMUTACIONES)
    for i in range(N_PERMUTACIONES):
        rs[i] = np.corrcoef(x, rng.permutation(y))[0, 1]
    p = float((np.abs(rs) >= abs(r_base)).mean())

    print("\n  G. Test de permutación "
          f"({N_PERMUTACIONES} barajadas al azar)")
    print(f"     r observado                : {r_base:.3f}")
    print(f"     r promedio al azar         : {rs.mean():+.3f}")
    print(f"     mayor |r| obtenido al azar : {np.abs(rs).max():.3f}")
    print(f"     veces que el azar igualó o superó al observado: "
          f"{int((np.abs(rs) >= abs(r_base)).sum())} de {N_PERMUTACIONES}  (p = {p:.3f})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("LECTURA")
    print("=" * 74)

    valores = [r_base,
               b["d_doc"].corr(b["d_est"]),
               cc["d_doc"].corr(cc["d_est"]),
               gr["d_doc"].corr(gr["d_est"]),
               corr_pond(c["d_doc"], c["d_est"], c["n_est_26"])]
    mn, mx = min(valores), max(valores)
    print(f"\n  El r se mueve entre {mn:.3f} y {mx:.3f} según la especificación.")

    if mn > 0.20:
        print("""
  El resultado AGUANTA. Sacar los centros extremos, quedarse solo con los
  grandes o ponderar por alumnos no lo desarma: la relación no la sostiene
  un puñado de centros rotos. Se puede defender con confianza.""")
    elif mn > 0.10:
        print("""
  El resultado se DEBILITA pero no desaparece. Hay algo de contribución de
  los centros extremos. Conviene presentar el número de la especificación
  más exigente, no el más favorable: es más chico pero es el que aguanta
  cualquier pregunta.""")
    else:
        print("""
  El resultado NO aguanta: al sacar los centros extremos la relación se
  cae. Eso significa que la correlación la sostenían unos pocos centros
  que se apagaron por completo, y no se puede afirmar arrastre general.""")

    if p < 0.01:
        print("""
  La permutación descarta el azar: barajando los datos nunca se llega a un
  r parecido. La correspondencia centro a centro es real.""")

    # dosis-respuesta en la muestra exigente
    if len(ex) > 100:
        ex = ex.copy()
        ex["grupo"] = pd.qcut(ex["d_doc"], 5,
                              labels=["Q1 (más cayó)", "Q2", "Q3", "Q4",
                                      "Q5 (menos cayó)"])
        dos = (ex.groupby("grupo", observed=True)
               .agg(centros=("d_est", "size"), d_doc=("d_doc", "mean"),
                    d_est=("d_est", "mean"),
                    alumnos=("n_est_26", "sum")).round(1).reset_index())
        print("\n  Dosis-respuesta en la muestra más exigente (B+C+D):")
        print(dos.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    resumen = pd.DataFrame({
        "especificacion": ["A base", "B sin colas", "C sin extremos",
                           "D grandes", "E ponderado"],
        "n": [len(c), len(b), len(cc), len(gr), len(c)],
        "r": [round(v, 3) for v in valores]})
    ruta = OUT / "robustez_arrastra.csv"
    resumen.to_csv(ruta, **CSV_KW)
    print(f"\n  -> {ruta}")


if __name__ == "__main__":
    main()