"""
centros.py — ¿La caída es generalizada entre centros, o está concentrada?

POR QUÉ ESTE CORTE ES DISTINTO
Todos los cortes anteriores (rubro, zona, sexo, ciclo, quintil) tienen pocas
categorías y se leen en una tabla. Los centros son miles: rankearlos no sirve
de nada. La pregunta útil es otra, y es la que ninguna de las otras vistas
puede responder:

    ¿TODOS los centros bajaron un poco, o la mayoría se mantuvo y un grupo
    chico se derrumbó?

Cambia por completo la recomendación:
  - Si la distribución está centrada en -7 pp con poca dispersión -> el problema
    es SISTÉMICO (algo de la plataforma, del calendario, de la propuesta
    pedagógica). No sirve intervenir centro por centro.
  - Si la mediana es ~0 y hay una cola larga de centros que colapsaron -> el
    problema es LOCALIZADO. Ahí sí conviene identificar esos centros e
    intervenir puntualmente.

Por eso el foco está en la DISTRIBUCIÓN y en la CONCENTRACIÓN (curva de Pareto),
no en un ranking.

TRES PRECAUCIONES (las mismas que venimos aplicando)
  1. n MÍNIMO. Un centro con 8 alumnos puede pasar de 50% a 12% sin que
     signifique nada. Se exige MIN_N matriculados en AMBOS años.
  2. CENTROS PRESENTES EN LOS DOS AÑOS. Si un centro aparece solo en uno, no
     hay comparación posible (cierre, apertura, o recodificación).
  3. VOLUMEN ADEMÁS DE PORCENTAJE. Igual que en el análisis por ciclo:
     usuarios_perdidos = n_2026 * (tasa_2025 - tasa_2026) / 100, a matrícula
     2026 fija, para aislar comportamiento de matrícula.

SALIDA
  Por consola: diagnóstico de distribución, curva de concentración, peores y
  mejores centros. CSV en data/processed/metricas/centros_<poblacion>.csv
"""

import sys
import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")

POBLACIONES = ["estudiantes", "docentes"]
MESES = ["Dias4", "Dias5", "Dias6"]

# Umbral de intensidad. 10 días es el que mejor separa la señal (ver
# metricas_acceso.py sección 5: la caída se profundiza con el umbral).
UMBRAL = 10

# Matriculados mínimos en cada año para que el % del centro sea creíble.
MIN_N = {"estudiantes": 30, "docentes": 10}

# Candidatas a columna identificadora de centro. Se busca por coincidencia
# parcial; "tipo_centro" se excluye a propósito (es una categoría, no un ID).
PISTAS_CENTRO = ["centro", "escuela", "liceo", "institucion", "local"]
EXCLUIR = {"tipo_centro", "tipo_de_centro"}

CSV_KW = dict(index=False, sep=";", decimal=",")


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def detectar_columna_centro(df: pd.DataFrame):
    """Busca la columna que identifica al centro. Devuelve None si no hay."""
    candidatas = []
    for c in df.columns:
        if c in EXCLUIR:
            continue
        if any(p in c for p in PISTAS_CENTRO):
            candidatas.append(c)

    if not candidatas:
        return None

    # Si hay varias, quedarse con la de mayor cardinalidad: un ID de centro
    # tiene cientos/miles de valores distintos, una categoría tiene 5.
    candidatas.sort(key=lambda c: df[c].nunique(), reverse=True)
    return candidatas[0]


def cargar(pob: str):
    """Carga 2025+2026 con dias_total. Devuelve (df, columna_centro)."""
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{pob}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()

        cols = [norm(c) for c in MESES]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dias_total"] = df[cols].sum(axis=1, min_count=1)
        df["anio"] = anio
        partes.append(df)

    df = pd.concat(partes, ignore_index=True)
    return df, detectar_columna_centro(df)


def tabla_centros(df: pd.DataFrame, col: str, pob: str) -> pd.DataFrame:
    """Una fila por centro con tasas de ambos años, var_pp y usuarios perdidos."""
    d = df.copy()
    d["ok"] = d["dias_total"] >= UMBRAL

    g = d.groupby([col, "anio"], dropna=False).agg(
        n=("ok", "size"), acc=("ok", "sum")
    ).reset_index()
    g["tasa"] = 100 * g["acc"] / g["n"]

    piv = g.pivot_table(index=col, columns="anio", values=["n", "acc", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    # Solo centros presentes y con tamaño suficiente en LOS DOS años.
    m = MIN_N[pob]
    antes = len(piv)
    piv = piv[(piv["n_2025"].fillna(0) >= m) & (piv["n_2026"].fillna(0) >= m)].copy()
    print(f"  centros totales: {antes:,} -> {len(piv):,} comparables "
          f"(presentes en ambos años y con n >= {m})")

    piv["var_pp"] = (piv["tasa_2026"] - piv["tasa_2025"]).round(1)
    piv["var_rel_%"] = (100 * (piv["tasa_2026"] - piv["tasa_2025"])
                        / piv["tasa_2025"]).round(1)
    piv["perdidos"] = (piv["n_2026"] * (piv["tasa_2025"] - piv["tasa_2026"]) / 100).round(0)
    for c in ("tasa_2025", "tasa_2026"):
        piv[c] = piv[c].round(1)
    return piv.rename(columns={col: "centro"})


def histograma(serie: pd.Series, ancho=5, escala=60):
    """Histograma de texto para ver la forma de la distribución en consola."""
    lo = int((serie.min() // ancho) * ancho)
    hi = int((serie.max() // ancho + 1) * ancho)
    bins = list(range(lo, hi + ancho, ancho))
    cortes = pd.cut(serie, bins=bins, right=False)
    conteo = cortes.value_counts().sort_index()
    tope = conteo.max() or 1
    for intervalo, c in conteo.items():
        barra = "#" * int(escala * c / tope)
        marca = "  <-- sin cambio" if intervalo.left <= 0 < intervalo.right else ""
        print(f"  {intervalo.left:>4.0f} a {intervalo.right:>4.0f} pp | "
              f"{c:>5,} {barra}{marca}")


def diagnostico(t: pd.DataFrame, pob: str):
    """Lo importante: ¿caída generalizada o concentrada?"""
    v = t["var_pp"].dropna()

    print("\n" + "-" * 70)
    print("A. DISTRIBUCIÓN DE LA CAÍDA ENTRE CENTROS")
    print("-" * 70)
    print(f"  centros analizados : {len(v):,}")
    print(f"  mediana            : {v.median():.1f} pp")
    print(f"  media              : {v.mean():.1f} pp")
    print(f"  percentil 10 / 90  : {v.quantile(.10):.1f} pp  /  {v.quantile(.90):.1f} pp")
    print(f"  centros que BAJAN  : {100*(v < 0).mean():.1f}%")
    print(f"  bajan más de 10 pp : {100*(v < -10).mean():.1f}%")
    print(f"  centros que SUBEN  : {100*(v > 0).mean():.1f}%")
    print()
    histograma(v)

    print("\n  CÓMO LEERLO:")
    if (v < 0).mean() > 0.70 and v.median() < -3:
        print("  La mayoría de los centros baja y la mediana está lejos de 0.")
        print("  -> La caída es GENERALIZADA (sistémica). Intervenir centro por")
        print("     centro no alcanza: el problema es transversal al sistema.")
    elif (v < -10).mean() > 0.20:
        print("  Hay una cola grande de centros con caídas fuertes.")
        print("  -> Convive un componente sistémico con un grupo de centros")
        print("     especialmente golpeados que merecen atención focalizada.")
    else:
        print("  La mediana está cerca de 0 y la caída se explica por una cola.")
        print("  -> La caída es LOCALIZADA. Vale la pena identificar esos centros.")

    # ---- concentración (Pareto) -----------------------------------------
    print("\n" + "-" * 70)
    print("B. CONCENTRACIÓN: ¿cuántos centros explican la caída total?")
    print("-" * 70)
    p = t[t["perdidos"] > 0].sort_values("perdidos", ascending=False)
    total = p["perdidos"].sum()
    if total <= 0:
        print("  (sin usuarios perdidos netos)")
        return
    p = p.assign(acum=p["perdidos"].cumsum())
    for frac in (0.05, 0.10, 0.25, 0.50):
        k = max(1, int(len(p) * frac))
        print(f"  el {frac:>4.0%} de centros con peor caída ({k:>5,} centros) "
              f"explica el {100*p['acum'].iloc[k-1]/total:>5.1f}% de la caída total")
    print(f"\n  total de {pob} que dejaron de usar CREA de forma sostenida: "
          f"{total:,.0f}")

    # ---- extremos --------------------------------------------------------
    cols = ["centro", "n_2025", "n_2026", "tasa_2025", "tasa_2026", "var_pp", "perdidos"]
    print("\n" + "-" * 70)
    print("C. 15 CENTROS QUE MÁS USUARIOS PERDIERON (volumen, no %)")
    print("-" * 70)
    print(p.head(15)[cols].to_string(index=False))

    print("\n" + "-" * 70)
    print("D. 15 CENTROS QUE MEJORARON (contraejemplos: ¿qué hicieron distinto?)")
    print("-" * 70)
    mejores = t.nlargest(15, "var_pp")
    print(mejores[cols].to_string(index=False))


def main():
    pd.set_option("display.width", 220)

    for pob in POBLACIONES:
        print("\n" + "=" * 70)
        print(f"USO POR CENTRO — {pob.upper()}  (umbral >= {UMBRAL} días)")
        print("=" * 70)

        df, col = cargar(pob)

        if col is None:
            print("\n  NO se encontró una columna que identifique al centro.")
            print("  Columnas disponibles en los datos:\n")
            for c in sorted(df.columns):
                print(f"    - {c}  ({df[c].nunique():,} valores distintos)")
            print("\n  Si alguna de estas es el centro, agregala a PISTAS_CENTRO")
            print("  arriba en el script y volvé a correr.")
            continue

        print(f"  columna de centro detectada: '{col}' "
              f"({df[col].nunique():,} centros distintos)")

        t = tabla_centros(df, col, pob)
        if t.empty:
            print("  No quedaron centros comparables. Revisá MIN_N.")
            continue

        diagnostico(t, pob)

        OUT.mkdir(parents=True, exist_ok=True)
        ruta = OUT / f"centros_{pob}.csv"
        t.sort_values("perdidos", ascending=False).to_csv(ruta, **CSV_KW)
        print(f"\n  -> guardado {ruta}")


if __name__ == "__main__":
    main()