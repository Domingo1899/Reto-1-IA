"""
auditoria.py — Detector automático de problemas de calidad de datos.

CÓMO FUNCIONA
No busca "errores" en abstracto: aplica REGLAS. Todo lo que viola una regla
es un hallazgo. Cada hallazgo dice: qué tabla, qué columna, cuántas filas,
y ejemplos reales de los valores ofensores.

SEVERIDADES
  ALTA  -> rompe el análisis o falsea números. Hay que resolverlo.
  MEDIA -> distorsiona resultados. Decidir y documentar.
  BAJA  -> informativo.

Cada hallazgo ALTA necesita una decisión escrita en docs/decisiones.md.
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida") ## la carpeta de la cual lee, es el cambio que hicimos para leer los datos neuvos 
SALIDA = Path("reports/auditoria")

# ---------------------------------------------------------------- LAS REGLAS
# Rango válido de días de acceso, según los días que tiene cada mes.
RANGOS_DIAS = {
    "Dias4": (0, 30),   # abril
    "Dias5": (0, 31),   # mayo
    "Dias6": (0, 30),   # junio
}

# Valores que un campo PUEDE tomar. Si aparece otro, es un error.
# (Los comparamos normalizados, así que 'f', 'F' y ' F ' cuentan igual.)
DOMINIOS = {
    "Sexo": {"f", "m"},
    "Zona": {"rural", "urbana"},
}

# Columnas que deberían tener quintiles del 1 al 5.
QUINTILES = ["Contexto Sociocultural", "CONTEXTO", "IvsMedia", "IVSMEDIA"]

# Textos que significan "vacío" pero que pandas lee como texto normal.
NULOS_DISFRAZADOS = {
    "", "-", "--", "s/d", "sd", "n/a", "na", "nan", "null", "none",
    "sin dato", "sin datos", ".", "?",
}

hallazgos = []  # acá se va acumulando todo


def anotar(tabla, severidad, columna, detalle, filas=0, ejemplos=""):
    hallazgos.append({
        "tabla": tabla, "severidad": severidad, "columna": columna,
        "detalle": detalle, "filas_afectadas": filas, "ejemplos": ejemplos,
    })


def norm(v) -> str:
    """'  MONTEVÍDEO ' -> 'montevideo'. Sirve para ver si dos textos son 'el mismo'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def ejemplos(valores, n=5):
    return " | ".join(repr(str(v)) for v in list(dict.fromkeys(valores))[:n])


# ------------------------------------------------------------------ CHEQUEOS

def chequear(df: pd.DataFrame, tabla: str):
    total = len(df)
    print(f"\n{'=' * 70}\n{tabla}  ({total:,} filas x {df.shape[1]} columnas)\n{'=' * 70}")

    # --- 1. FILAS DUPLICADAS ---
    n = int(df.duplicated().sum())
    if n:
        anotar(tabla, "ALTA", "-",
               f"{n:,} filas idénticas a otra ({100*n/total:.2f}%). Error de exportación.", n)

    # --- 2. NULOS (reales + disfrazados) ---
    for col in df.columns:
        vacios = df[col].map(lambda v: norm(v) in NULOS_DISFRAZADOS) | df[col].isna()
        n = int(vacios.sum())
        if n:
            pct = 100 * n / total
            sev = "ALTA" if pct >= 20 else ("MEDIA" if pct >= 1 else "BAJA")
            # OJO: en Contexto/IvsMedia los nulos son ESPERADOS (ver decisiones D2)
            nota = ""
            if col in QUINTILES:
                sev = "BAJA"
                nota = " [ESPERADO: solo aplica a un subsistema, ver D2]"
            anotar(tabla, sev, col, f"{n:,} valores vacíos ({pct:.1f}%){nota}", n,
                   ejemplos(df.loc[vacios, col].dropna().unique()))

    # --- 3. DÍAS DE ACCESO: fuera de rango, negativos, no numéricos ---
    for col, (lo, hi) in RANGOS_DIAS.items():
        if col not in df.columns:
            continue
        serie = df[col]
        num = pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False), errors="coerce")

        # texto donde debería haber números
        no_num = num.isna() & serie.notna() & (serie.map(norm) != "")
        if no_num.sum():
            anotar(tabla, "ALTA", col,
                   f"{int(no_num.sum()):,} valores NO numéricos en una columna de días.",
                   int(no_num.sum()), ejemplos(serie[no_num].unique()))

        # negativos
        neg = num < 0
        if neg.sum():
            anotar(tabla, "ALTA", col,
                   f"{int(neg.sum()):,} días negativos: imposible.",
                   int(neg.sum()), ejemplos(num[neg].unique()))

        # fuera del rango del mes
        fuera = ((num < lo) | (num > hi)) & num.notna()
        if fuera.sum():
            anotar(tabla, "ALTA", col,
                   f"{int(fuera.sum()):,} valores fuera del rango [{lo},{hi}]. Máximo: {num.max():.0f}",
                   int(fuera.sum()), ejemplos(num[fuera].unique()))

    # --- 4. DOMINIOS: valores que no deberían existir ---
    for col, permitidos in DOMINIOS.items():
        real = next((c for c in df.columns if norm(c) == norm(col)), None)  # Zona o ZONA
        if real is None:
            continue
        valores = df[real].dropna()
        invalidos = valores[~valores.map(norm).isin(permitidos | {""})]
        if len(invalidos):
            anotar(tabla, "ALTA", real,
                   f"{len(invalidos):,} valores fuera del dominio permitido {sorted(permitidos)}.",
                   len(invalidos), ejemplos(invalidos.unique()))

    # --- 5. QUINTILES: deben ir de 1 a 5 ---
    for col in QUINTILES:
        real = next((c for c in df.columns if norm(c) == norm(col)), None)
        if real is None:
            continue
        num = pd.to_numeric(df[real], errors="coerce")
        fuera = ((num < 1) | (num > 5)) & num.notna()
        if fuera.sum():
            anotar(tabla, "ALTA", real,
                   f"{int(fuera.sum()):,} quintiles fuera del rango 1-5.",
                   int(fuera.sum()), ejemplos(num[fuera].unique()))

    # --- 6. CATEGORÍAS ESCRITAS DE VARIAS FORMAS ---
    # 'Montevideo' y 'MONTEVIDEO' se cuentan como 2 departamentos distintos.
    for col in df.columns:
        valores = df[col].dropna().astype(str)
        if valores.empty or valores.nunique() > 60:   # no es categórica
            continue
        grupos = {}
        for v in valores.unique():
            grupos.setdefault(norm(v), set()).add(v)
        colisiones = {k: v for k, v in grupos.items() if len(v) > 1}
        if colisiones:
            afectadas = {x for g in colisiones.values() for x in g}
            n = int(valores.isin(afectadas).sum())
            anotar(tabla, "ALTA", col,
                   f"{len(colisiones)} categorías escritas de formas distintas. "
                   f"Sin normalizar, los conteos por '{col}' quedan MAL.",
                   n, ejemplos(sorted(next(iter(colisiones.values())))))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivos = sorted(CARPETA.glob("*.parquet"))
    if not archivos:
        print(f"No hay .parquet en {CARPETA}. Corré antes convertir_a_parquet.py")
        return

    tablas = {}
    for ruta in archivos:
        df = pd.read_parquet(ruta)
        tablas[ruta.stem] = df
        chequear(df, ruta.stem)

    # ---- resultados ----
    res = pd.DataFrame(hallazgos)
    if res.empty:
        print("\nNo se detectaron problemas. (Revisar igual: puede faltar una regla.)")
        return

    orden = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}
    res = res.sort_values(by="severidad", key=lambda s: s.map(orden)).reset_index(drop=True)

    print(f"\n\n{'=' * 70}\nRESUMEN\n{'=' * 70}")
    for sev in ["ALTA", "MEDIA", "BAJA"]:
        print(f"  {sev:>5}: {(res.severidad == sev).sum():>3} hallazgos")

    print(f"\nHALLAZGOS DE SEVERIDAD ALTA:")
    for _, r in res[res.severidad == "ALTA"].iterrows():
        print(f"  [{r.tabla:<18}] {str(r.columna)[:24]:<24} {r.detalle}")
        if r.ejemplos:
            print(f"   {'':<19} ejemplos: {r.ejemplos[:90]}")

    ruta_csv = SALIDA / "hallazgos_entrega-01.csv"
    res.to_csv(ruta_csv, index=False)
    print(f"\nCatálogo completo guardado en: {ruta_csv}")
    print("Próximo paso: cada hallazgo ALTA necesita una decisión en docs/decisiones.md")


if __name__ == "__main__":
    main()